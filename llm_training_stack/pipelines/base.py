"""Base training pipeline with lifecycle management, telemetry, and checkpointing."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterator

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback

from llm_training_stack.config.schema import TrainingJobConfig
from llm_training_stack.provenance.manifest import RunManifest
from llm_training_stack.provenance.event_logger import StructuredEventLogger
from llm_training_stack.checkpoints.manager import CheckpointManager
from llm_training_stack.checkpoints.resume import ResumeManager


def is_explicit_mock_dataset(name: str) -> bool:
    """Checks whether the dataset name explicitly identifies as a synthetic/mock test dataset."""
    if not name:
        return False
    clean = name.strip().lower()
    return (
        clean in ("synthetic", "mock", "dummy", "mock_dataset", "synthetic_test", "mock_sft", "mock_cpt", "mock_dpo")
        or clean.startswith("synthetic:")
        or clean.startswith("mock:")
        or clean.startswith("dummy:")
    )


def load_strict_dataset(dataset_name_or_path: str, split: str = "train", mock_factory=None):
    """Enforce strict dataset loading: raise RuntimeError on failed dataset loads

    unless explicitly identifying as mock/synthetic test datasets.
    """
    from datasets import load_dataset

    if is_explicit_mock_dataset(dataset_name_or_path):
        if mock_factory is not None:
            return mock_factory()
        raise RuntimeError(f"Mock dataset requested for '{dataset_name_or_path}' but no mock factory provided.")

    path_obj = Path(dataset_name_or_path)
    try:
        if path_obj.exists() and path_obj.is_file():
            ext = path_obj.suffix.lower().lstrip(".")
            if ext == "jsonl":
                import json
                with open(path_obj, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f if line.strip()]
                    if not data:
                        raise ValueError(f"Dataset file '{dataset_name_or_path}' contains no samples (empty file).")
                    return data
            elif ext == "json":
                import json
                with open(path_obj, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data = data if isinstance(data, list) else [data]
                    if not data:
                        raise ValueError(f"Dataset file '{dataset_name_or_path}' contains no samples (empty file).")
                    return data
            elif ext == "parquet":
                return load_dataset("parquet", data_files=str(path_obj), split="train")
            elif ext == "csv":
                return load_dataset("csv", data_files=str(path_obj), split="train")
            elif ext in ("txt", "text"):
                return load_dataset("text", data_files=str(path_obj), split="train")
        return load_dataset(dataset_name_or_path, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name_or_path}': {e}. "
            "To use synthetic data for testing, explicitly specify 'synthetic' or 'mock'."
        ) from e


class BaseTrainingPipeline(ABC):
    """Abstract orchestrator for LLM training pipelines."""

    def __init__(self, config: TrainingJobConfig):
        self.config = config
        self.output_dir = Path(config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device()
        self.manifest = RunManifest.create(self.config)
        self.manifest_path = RunManifest.save(self.manifest, self.output_dir)
        self.event_logger = StructuredEventLogger(self.output_dir / "events.jsonl")
        self.checkpoint_manager = CheckpointManager(
            self.output_dir, save_total_limit=self.config.logging.save_total_limit
        )

        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.is_peft = False

    def _resolve_device(self) -> torch.device:
        target = self.config.hardware.target_device
        if target == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Configured target_device='cuda' but CUDA is not available on this host. "
                    "Silent CPU fallback is rejected per project contract."
                )
            return torch.device("cuda:0")
        elif target == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def apply_peft(self) -> None:
        """Injects LoRA adapters into model weights according to PeftConfig."""
        from peft import LoraConfig, get_peft_model

        peft_cfg = self.config.peft
        lora_config = LoraConfig(
            r=peft_cfg.r,
            lora_alpha=peft_cfg.lora_alpha,
            lora_dropout=peft_cfg.lora_dropout,
            target_modules=peft_cfg.target_modules,
            bias=peft_cfg.bias,
            task_type=peft_cfg.task_type,
            modules_to_save=peft_cfg.modules_to_save,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.is_peft = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        self.manifest["peft_stats"] = {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": round(100 * trainable_params / all_params, 3),
            "target_modules": peft_cfg.target_modules,
            "lora_rank": peft_cfg.r,
            "lora_alpha": peft_cfg.lora_alpha,
        }

    def setup_model_and_tokenizer(self) -> None:
        model_name = self.config.model.model_name_or_path
        tok_name = self.config.model.tokenizer_name_or_path or model_name
        model_rev = self.config.model.revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_name,
            revision=model_rev,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "auto": None,
        }
        torch_dtype = dtype_map.get(self.config.model.torch_dtype, None)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_rev,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.model.to(self.device)

        # Apply PEFT if configured (orthogonal to task type: CPT, SFT, DPO)
        if self.config.peft is not None:
            self.apply_peft()

    def setup_optimizer(self) -> None:
        import math
        if self.model is None:
            raise RuntimeError("Model must be initialized via setup_model_and_tokenizer() before setup_optimizer().")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_cfg = self.config.optimizer
        opt_type = opt_cfg.optimizer_type.lower()

        if opt_type in ("adamw_torch", "adamw_hf", "adamw"):
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt_cfg.learning_rate,
                betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                eps=opt_cfg.adam_epsilon,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=opt_cfg.learning_rate,
                betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                eps=opt_cfg.adam_epsilon,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_type == "adamw_torch_fused":
            try:
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=opt_cfg.learning_rate,
                    betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                    eps=opt_cfg.adam_epsilon,
                    weight_decay=opt_cfg.weight_decay,
                    fused=True,
                )
            except Exception:
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=opt_cfg.learning_rate,
                    betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
                    eps=opt_cfg.adam_epsilon,
                    weight_decay=opt_cfg.weight_decay,
                )
        else:
            raise ValueError(
                f"Unsupported optimizer_type '{opt_cfg.optimizer_type}'. "
                f"Supported types: adamw_torch, adamw_hf, adamw, adam, sgd, adamw_torch_fused."
            )

        grad_accum = self.config.hardware.gradient_accumulation_steps
        if self.config.max_steps is not None:
            max_micro_steps = self.config.max_steps
        else:
            sample_count = self.config.dataset.train_sample_limit or 100
            batch_size = max(1, self.config.hardware.per_device_train_batch_size)
            max_micro_steps = int(self.config.epochs * max(1, sample_count // batch_size))

        total_opt_steps = max(1, max_micro_steps // grad_accum)
        warmup_steps = int(total_opt_steps * opt_cfg.warmup_ratio)

        sched_type = opt_cfg.lr_scheduler_type.lower()
        if sched_type == "linear":
            def lr_lambda(current_opt_step: int):
                if current_opt_step < warmup_steps:
                    return float(current_opt_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_opt_steps - current_opt_step) / float(max(1, total_opt_steps - warmup_steps)))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif sched_type == "cosine":
            def lr_lambda(current_opt_step: int):
                if current_opt_step < warmup_steps:
                    return float(current_opt_step) / float(max(1, warmup_steps))
                progress = float(current_opt_step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif sched_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        elif sched_type == "constant_with_warmup":
            def lr_lambda(current_opt_step: int):
                if current_opt_step < warmup_steps:
                    return float(current_opt_step) / float(max(1, warmup_steps))
                return 1.0
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type '{opt_cfg.lr_scheduler_type}'. "
                f"Supported types: linear, cosine, constant, constant_with_warmup."
            )

    @abstractmethod
    def get_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yields training batches with tensors already formatted."""
        pass

    def build_training_arguments(self, interrupt_after_step: Optional[int] = None) -> TrainingArguments:
        """Converts TrainingJobConfig into upstream transformers TrainingArguments."""
        if interrupt_after_step is not None:
            max_steps = interrupt_after_step
            num_train_epochs = 1.0
        elif self.config.max_steps is not None:
            max_steps = self.config.max_steps
            num_train_epochs = 1.0
        else:
            max_steps = -1
            num_train_epochs = float(self.config.epochs)

        optim_map = {
            "adamw_torch": "adamw_torch",
            "adamw_hf": "adamw_hf",
            "adamw": "adamw_torch",
            "sgd": "sgd",
            "adamw_torch_fused": "adamw_torch_fused",
        }
        opt_type = self.config.optimizer.optimizer_type.lower()
        if opt_type not in optim_map:
            raise ValueError(f"Unsupported optimizer_type '{self.config.optimizer.optimizer_type}' for upstream Trainer.")
        optim = optim_map[opt_type]

        effective_logging_steps = self.config.logging.logging_steps
        if max_steps is not None and max_steps > 0:
            effective_logging_steps = max(1, min(effective_logging_steps, max_steps))

        return TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.config.hardware.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.hardware.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.hardware.gradient_accumulation_steps,
            learning_rate=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
            warmup_ratio=self.config.optimizer.warmup_ratio,
            max_grad_norm=self.config.optimizer.max_grad_norm,
            lr_scheduler_type=self.config.optimizer.lr_scheduler_type,
            optim=optim,
            logging_steps=effective_logging_steps,
            save_steps=self.config.logging.save_steps,
            save_total_limit=self.config.logging.save_total_limit,
            seed=self.config.seed,
            fp16=(self.config.hardware.mixed_precision == "fp16"),
            bf16=(self.config.hardware.mixed_precision == "bf16"),
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            dataloader_num_workers=self.config.hardware.dataloader_num_workers,
            use_cpu=(self.device.type == "cpu"),
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            report_to="none",
            save_strategy="steps",
            logging_strategy="steps",
            remove_unused_columns=False,
        )

    def build_trainer(self, callbacks: Optional[list] = None) -> Any:
        """Constructs upstream transformers.Trainer / trl.SFTTrainer / trl.DPOTrainer instance."""
        raise NotImplementedError("Subclasses must implement build_trainer() for upstream execution.")

    def execute_upstream_trainer(
        self,
        resume_from: Optional[Path] = None,
        interrupt_after_step: Optional[int] = None,
        cancel_check: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes training using genuine upstream transformers.Trainer / trl Trainer backend."""
        self.setup_model_and_tokenizer()
        cancel_cb = CancellationCallback(self.output_dir, cancel_check=cancel_check)
        callbacks = [
            EventLoggingCallback(self.event_logger, self.config.max_steps or 50),
            cancel_cb,
        ]
        interrupt_cb = None
        if interrupt_after_step is not None:
            interrupt_cb = StepInterruptCallback(interrupt_after_step)
            callbacks.append(interrupt_cb)

        trainer = self.build_trainer(callbacks=callbacks)
        start_time = time.time()
        train_res = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

        is_interrupted = False
        if interrupt_cb is not None and getattr(interrupt_cb, "was_interrupted", False):
            is_interrupted = True

        is_cancelled = (
            cancel_cb.was_cancelled
            or (self.output_dir / "cancel.token").exists()
            or bool(cancel_check and cancel_check())
        )

        total_steps = getattr(train_res, "global_step", self.config.max_steps or 0)
        final_loss = getattr(train_res, "training_loss", None)

        final_cp = self.checkpoint_manager.save_checkpoint(
            step=total_steps,
            epoch=1.0,
            model=self.model,
            optimizer=trainer.optimizer if hasattr(trainer, "optimizer") else None,
            lr_scheduler=trainer.lr_scheduler if hasattr(trainer, "lr_scheduler") else None,
            loss=final_loss or 0.0,
            dataloader_index=total_steps,
            is_peft=self.is_peft,
        )

        try:
            trainer.save_model(str(final_cp))
        except Exception as e:
            logger.warning(f"Could not execute trainer.save_model: {e}")

        try:
            trainer.save_state()
            output_state = self.output_dir / "trainer_state.json"
            if output_state.exists():
                import shutil
                shutil.copy2(output_state, final_cp / "trainer_state.json")
            if hasattr(trainer, "state"):
                trainer.state.save_to_json(str(final_cp / "trainer_state.json"))
        except Exception as e:
            logger.warning(f"Could not persist trainer_state.json into checkpoint: {e}")

        total_duration = time.time() - start_time
        if is_cancelled:
            status = "CANCELLED"
        elif is_interrupted:
            status = "INTERRUPTED"
        else:
            status = "COMPLETED"

        summary = {
            "status": status,
            "total_steps": total_steps,
            "total_duration_sec": round(total_duration, 2),
            "final_loss": final_loss,
            "final_checkpoint": str(final_cp),
            "backend": "transformers",
        }
        self.manifest["status"] = status
        self.manifest["final_metrics"] = summary
        RunManifest.save(self.manifest, self.output_dir)
        self.event_logger.close()
        return summary

    def train_direct(
        self,
        resume_from: Optional[Path] = None,
        interrupt_after_step: Optional[int] = None,
        cancel_check: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Executes direct PyTorch training with gradient accumulation, resume, and finite dataset flush."""
        self.setup_model_and_tokenizer()
        self.setup_optimizer()

        start_step = 0
        start_epoch = 0.0
        if resume_from is not None:
            res_info = ResumeManager.resume_into(
                Path(resume_from), self.model, self.optimizer, self.lr_scheduler
            )
            start_step = res_info["step"]
            start_epoch = res_info["epoch"]

        self.model.train()
        global_step = start_step
        grad_accum = self.config.hardware.gradient_accumulation_steps
        optimizer_step = start_step // grad_accum
        max_steps = self.config.max_steps or 50

        batch_iterator = self.get_batches()
        if start_step > 0:
            import itertools
            batch_iterator = itertools.islice(batch_iterator, start_step, None)

        accumulated_loss = 0.0
        loss_history = []

        self.optimizer.zero_grad()
        start_time = time.time()
        is_cancelled = False

        for batch in batch_iterator:
            if (self.output_dir / "cancel.token").exists() or bool(cancel_check and cancel_check()):
                logger.info(f"Cancellation token detected at step {global_step}; stopping training.")
                is_cancelled = True
                break
            if global_step >= max_steps:
                break
            if interrupt_after_step is not None and global_step >= interrupt_after_step:
                break

            # Move batch tensors to device
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            step_start = time.time()

            outputs = self.model(**batch_device)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            accumulated_loss += loss.item()

            if (global_step + 1) % grad_accum == 0 or (global_step + 1) >= max_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.optimizer.max_grad_norm,
                )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                optimizer_step += 1

                step_time_ms = (time.time() - step_start) * 1000
                current_lr = self.optimizer.param_groups[0]["lr"]
                tokens_count = batch_device["input_ids"].numel()
                tokens_per_sec = tokens_count / max(step_time_ms / 1000.0, 1e-4)

                avg_loss = accumulated_loss / grad_accum
                loss_history.append(avg_loss)
                accumulated_loss = 0.0

                self.event_logger.log_step(
                    step=global_step + 1,
                    loss=avg_loss,
                    lr=current_lr,
                    epoch=(global_step + 1) / max(1, max_steps),
                    step_time_ms=step_time_ms,
                    tokens_per_sec=tokens_per_sec,
                )

                # Periodic Checkpoint strictly at optimizer step boundaries (unaccumulated gradients == 0)
                if (global_step + 1) % self.config.logging.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        step=global_step + 1,
                        epoch=(global_step + 1) / max(1, max_steps),
                        model=self.model,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        loss=avg_loss,
                        dataloader_index=global_step + 1,
                        is_peft=self.is_peft,
                    )

            global_step += 1

        # Flush pending unaccumulated gradients upon dataset exhaustion so microbatch gradients are never dropped
        unaccumulated_microbatches = global_step % grad_accum
        if not is_cancelled and unaccumulated_microbatches != 0 and accumulated_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.optimizer.max_grad_norm,
            )
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            optimizer_step += 1

            avg_loss = accumulated_loss / unaccumulated_microbatches
            loss_history.append(avg_loss)
            accumulated_loss = 0.0

            self.event_logger.log_step(
                step=global_step,
                loss=avg_loss,
                lr=self.optimizer.param_groups[0]["lr"],
                epoch=global_step / max(1, max_steps),
                step_time_ms=0.0,
                tokens_per_sec=0.0,
            )

        # Handle cancellation
        if is_cancelled:
            cancel_cp = self.checkpoint_manager.save_checkpoint(
                step=global_step,
                epoch=global_step / max(1, max_steps),
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                loss=loss_history[-1] if loss_history else 0.0,
                dataloader_index=global_step,
                is_peft=self.is_peft,
            )
            total_duration = time.time() - start_time
            summary = {
                "status": "CANCELLED",
                "total_steps": global_step,
                "total_duration_sec": round(total_duration, 2),
                "final_loss": loss_history[-1] if loss_history else None,
                "final_checkpoint": str(cancel_cp),
                "backend": "torch",
            }
            self.manifest["status"] = "CANCELLED"
            self.manifest["final_metrics"] = summary
            RunManifest.save(self.manifest, self.output_dir)
            self.event_logger.close()
            return summary

        # Handle early interruption exit
        if interrupt_after_step is not None and global_step >= interrupt_after_step:
            int_cp = self.checkpoint_manager.save_checkpoint(
                step=global_step,
                epoch=global_step / max(1, max_steps),
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                loss=loss_history[-1] if loss_history else 0.0,
                dataloader_index=global_step,
                is_peft=self.is_peft,
            )
            self.event_logger.close()
            return {
                "status": "INTERRUPTED",
                "steps": global_step,
                "total_steps": global_step,
                "checkpoint": str(int_cp),
            }

        # Final Checkpoint
        final_cp = self.checkpoint_manager.save_checkpoint(
            step=global_step,
            epoch=1.0,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss=loss_history[-1] if loss_history else 0.0,
            dataloader_index=global_step,
            is_peft=self.is_peft,
        )

        total_duration = time.time() - start_time
        summary = {
            "status": "COMPLETED",
            "total_steps": global_step,
            "total_duration_sec": round(total_duration, 2),
            "final_loss": loss_history[-1] if loss_history else None,
            "final_checkpoint": str(final_cp),
            "backend": "torch",
        }

        self.manifest["status"] = "COMPLETED"
        self.manifest["final_metrics"] = summary
        RunManifest.save(self.manifest, self.output_dir)
        self.event_logger.close()

        return summary

    def train(
        self,
        resume_from: Optional[Path] = None,
        interrupt_after_step: Optional[int] = None,
        backend: Optional[str] = None,
        cancel_check: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Main training entry point dispatching to requested or auto-detected backend."""
        chosen_backend = backend or getattr(self.config, "backend", "direct")
        if chosen_backend in ("transformers", "upstream"):
            return self.execute_upstream_trainer(
                resume_from=resume_from,
                interrupt_after_step=interrupt_after_step,
                cancel_check=cancel_check,
            )
        return self.train_direct(
            resume_from=resume_from,
            interrupt_after_step=interrupt_after_step,
            cancel_check=cancel_check,
        )


class CancellationCallback(TrainerCallback):
    """Production callback that halts upstream Trainer immediately when cancellation is requested."""

    def __init__(self, output_dir: Path, cancel_check: Optional[Any] = None):
        self.output_dir = Path(output_dir)
        self.cancel_check = cancel_check
        self.was_cancelled = False

    def on_substep_end(self, args, state, control, **kwargs):
        cancel_token = self.output_dir / "cancel.token"
        if cancel_token.exists() or bool(self.cancel_check and self.cancel_check()):
            control.should_training_stop = True
            control.should_save = True
            self.was_cancelled = True

    def on_step_end(self, args, state, control, **kwargs):
        cancel_token = self.output_dir / "cancel.token"
        if cancel_token.exists() or bool(self.cancel_check and self.cancel_check()):
            control.should_training_stop = True
            control.should_save = True
            self.was_cancelled = True


class StepInterruptCallback(TrainerCallback):
    """Production callback that halts training at a designated step for deterministic interruption testing."""

    def __init__(self, interrupt_step: int):
        self.interrupt_step = interrupt_step
        self.was_interrupted = False

    def on_step_end(self, args, state, control, **kwargs):
        cancel_token = Path(args.output_dir) / "cancel.token" if hasattr(args, "output_dir") else None
        if (cancel_token and cancel_token.exists()) or (self.interrupt_step is not None and state.global_step >= self.interrupt_step):
            control.should_training_stop = True
            control.should_save = True
            self.was_interrupted = True


class EventLoggingCallback(TrainerCallback):
    """Streams training loss, learning rate, epoch, and throughput to StructuredEventLogger."""

    def __init__(self, event_logger: StructuredEventLogger, total_steps: int):
        self.event_logger = event_logger
        self.total_steps = max(1, total_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.event_logger.log_step(
                step=state.global_step,
                loss=float(logs.get("loss", 0.0)),
                lr=float(logs.get("learning_rate", 0.0)),
                epoch=float(state.epoch or (state.global_step / self.total_steps)),
                step_time_ms=0.0,
                tokens_per_sec=0.0,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if getattr(self.event_logger, "step_count", 0) == 0 and state.log_history:
            last_entry = state.log_history[-1]
            self.event_logger.log_step(
                step=state.global_step,
                loss=float(last_entry.get("loss", last_entry.get("train_loss", 0.0))),
                lr=float(last_entry.get("learning_rate", 0.0)),
                epoch=float(state.epoch or 1.0),
                step_time_ms=0.0,
                tokens_per_sec=0.0,
            )

