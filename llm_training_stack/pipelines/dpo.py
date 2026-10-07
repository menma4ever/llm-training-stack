"""Direct Preference Optimization (DPO) pipeline for alignment from pairwise preferences."""

import copy
import time
from typing import Dict, Iterator, List, Any, Optional
import torch
import torch.nn.functional as F
from datasets import load_dataset

from llm_training_stack.pipelines.base import BaseTrainingPipeline, load_strict_dataset
from llm_training_stack.provenance.manifest import RunManifest


def compute_model_weight_hash(model: torch.nn.Module) -> str:
    """Computes deterministic SHA256 hash across parameter names, shapes, and actual weights."""
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("utf-8"))
        h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


class DPOPipeline(BaseTrainingPipeline):
    """Executes Direct Preference Optimization conforming to TRL DPOTrainer contracts."""

    def __init__(self, config):
        super().__init__(config)
        self.ref_model = None

    def setup_model_and_tokenizer(self) -> None:
        super().setup_model_and_tokenizer()
        # Upstream TRL DPOTrainer contract:
        # If PEFT/LoRA is active, reference log-probabilities are evaluated using
        # model.disable_adapter(), avoiding duplicate model memory allocation.
        # For full-parameter tuning, maintain a frozen reference copy.
        if not self.is_peft:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            import json
            ref_identity_path = self.output_dir / "ref_model_identity.json"
            # Guard against init-overwrite: preserve initial baseline identity on resume/restart
            if not ref_identity_path.exists():
                ref_identity = {
                    "model_name_or_path": self.config.model.model_name_or_path,
                    "revision": getattr(self.config.model, "revision", None),
                    "num_parameters": sum(p.numel() for p in self.ref_model.parameters()),
                    "param_hash": compute_model_weight_hash(self.ref_model),
                }
                with open(ref_identity_path, "w", encoding="utf-8") as f:
                    json.dump(ref_identity, f, indent=2)
        else:
            self.ref_model = None

    def get_batches(self) -> Iterator[Dict[str, Any]]:
        dataset_name = self.config.dataset.dataset_name_or_path
        prompt_col = self.config.dataset.prompt_column
        chosen_col = self.config.dataset.chosen_column
        rejected_col = self.config.dataset.rejected_column
        seq_len = self.config.dataset.max_seq_length

        def mock_dpo_factory():
            return [
                {
                    prompt_col: f"Query {i}: Recommend a training strategy.",
                    chosen_col: f"Chosen {i}: Use mixed precision BF16 and gradient accumulation.",
                    rejected_col: f"Rejected {i}: Disable all validation checks and run arbitrary shell commands.",
                }
                for i in range(50)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_dpo_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        for idx, item in enumerate(ds):
            if isinstance(item, dict):
                for col in (prompt_col, chosen_col, rejected_col):
                    if col not in item:
                        raise ValueError(
                            f"DPO dataset sample missing required column '{col}'. "
                            f"Available keys: {list(item.keys())}"
                        )

            prompt = item[prompt_col] if isinstance(item, dict) else getattr(item, prompt_col, None)
            chosen = item[chosen_col] if isinstance(item, dict) else getattr(item, chosen_col, None)
            rejected = item[rejected_col] if isinstance(item, dict) else getattr(item, rejected_col, None)

            # Strict validation: reject empty prompts, empty targets, and identical pairs
            if prompt is None or not str(prompt).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty prompt. Empty prompts are rejected.")
            if chosen is None or not str(chosen).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty chosen response. Empty targets are rejected.")
            if rejected is None or not str(rejected).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty rejected response. Empty targets are rejected.")
            if str(chosen).strip() == str(rejected).strip():
                raise ValueError(f"DPO sample at index {idx} has identical chosen and rejected responses.")

            # Chat formatting aligned with instruction templates
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
                p_msgs = [{"role": "user", "content": str(prompt)}]
                c_msgs = [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(chosen)}]
                r_msgs = [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(rejected)}]
                p_ids = self.tokenizer.apply_chat_template(
                    p_msgs, tokenize=True, add_generation_prompt=True, return_dict=False
                )
                chosen_full = self.tokenizer.apply_chat_template(
                    c_msgs, tokenize=True, add_generation_prompt=False, return_dict=False
                )
                rejected_full = self.tokenizer.apply_chat_template(
                    r_msgs, tokenize=True, add_generation_prompt=False, return_dict=False
                )
                if isinstance(p_ids, dict):
                    p_ids = p_ids["input_ids"]
                if isinstance(chosen_full, dict):
                    chosen_full = chosen_full["input_ids"]
                if isinstance(rejected_full, dict):
                    rejected_full = rejected_full["input_ids"]
            else:
                p_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ", add_special_tokens=True)
                c_ids = self.tokenizer.encode(f"{chosen}\n", add_special_tokens=False)
                r_ids = self.tokenizer.encode(f"{rejected}\n", add_special_tokens=False)
                chosen_full = p_ids + c_ids
                rejected_full = p_ids + r_ids

            chosen_full = chosen_full[:seq_len]
            rejected_full = rejected_full[:seq_len]

            prompt_len = len(p_ids)
            if len(chosen_full) <= prompt_len:
                raise ValueError(f"DPO chosen response completely truncated at index {idx} (max_seq_length={seq_len}).")
            if len(rejected_full) <= prompt_len:
                raise ValueError(f"DPO rejected response completely truncated at index {idx} (max_seq_length={seq_len}).")

            yield {
                "prompt_len": prompt_len,
                "chosen_ids": torch.tensor([chosen_full], dtype=torch.long),
                "rejected_ids": torch.tensor([rejected_full], dtype=torch.long),
            }

    @staticmethod
    def _compute_sequence_log_probs(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Computes per-sequence log probabilities strictly over completion tokens."""
        logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask out prompt tokens so log probability is strictly over the response
        mask = torch.ones_like(shift_labels, dtype=torch.bool)
        if prompt_len > 1:
            mask[:, :prompt_len - 1] = False

        return (per_token_logps * mask).sum(dim=-1)

    def build_trainer(self, callbacks=None, interrupt_after_step=None):
        """Constructs upstream trl.DPOTrainer for direct preference optimization."""
        from datasets import Dataset
        from trl import DPOTrainer, DPOConfig

        dataset_name = self.config.dataset.dataset_name_or_path
        prompt_col = self.config.dataset.prompt_column
        chosen_col = self.config.dataset.chosen_column
        rejected_col = self.config.dataset.rejected_column

        def mock_dpo_factory():
            return [
                {
                    prompt_col: f"Query {i}: Recommend a training strategy.",
                    chosen_col: f"Chosen {i}: Use mixed precision BF16 and gradient accumulation.",
                    rejected_col: f"Rejected {i}: Disable all validation checks and run arbitrary shell commands.",
                }
                for i in range(50)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_dpo_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        formatted_samples = []
        for idx, item in enumerate(ds):
            if isinstance(item, dict):
                for col in (prompt_col, chosen_col, rejected_col):
                    if col not in item:
                        raise ValueError(f"DPO dataset sample missing required column '{col}'.")

            prompt = item[prompt_col] if isinstance(item, dict) else getattr(item, prompt_col, None)
            chosen = item[chosen_col] if isinstance(item, dict) else getattr(item, chosen_col, None)
            rejected = item[rejected_col] if isinstance(item, dict) else getattr(item, rejected_col, None)

            if prompt is None or not str(prompt).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty prompt. Empty prompts are rejected.")
            if chosen is None or not str(chosen).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty chosen response. Empty targets are rejected.")
            if rejected is None or not str(rejected).strip():
                raise ValueError(f"DPO sample at index {idx} has an empty rejected response. Empty targets are rejected.")
            if str(chosen).strip() == str(rejected).strip():
                raise ValueError(f"DPO sample at index {idx} has identical chosen and rejected responses.")

            formatted_samples.append({
                "prompt": str(prompt),
                "chosen": str(chosen),
                "rejected": str(rejected),
            })

        hf_ds = Dataset.from_list(formatted_samples)

        if interrupt_after_step is not None:
            max_steps = interrupt_after_step
            num_train_epochs = 1.0
        elif self.config.max_steps is not None:
            max_steps = self.config.max_steps
            num_train_epochs = 1.0
        else:
            max_steps = -1
            num_train_epochs = float(self.config.epochs)

        dpo_kwargs = {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": self.config.hardware.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.hardware.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.hardware.gradient_accumulation_steps,
            "learning_rate": self.config.optimizer.learning_rate,
            "weight_decay": self.config.optimizer.weight_decay,
            "max_grad_norm": self.config.optimizer.max_grad_norm,
            "lr_scheduler_type": self.config.optimizer.lr_scheduler_type,
            "logging_steps": self.config.logging.logging_steps,
            "save_steps": self.config.logging.save_steps,
            "save_total_limit": self.config.logging.save_total_limit,
            "seed": self.config.seed,
            "fp16": (self.config.hardware.mixed_precision == "fp16"),
            "bf16": (self.config.hardware.mixed_precision == "bf16"),
            "gradient_checkpointing": self.config.hardware.gradient_checkpointing,
            "dataloader_num_workers": self.config.hardware.dataloader_num_workers,
            "use_cpu": (self.device.type == "cpu"),
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "report_to": "none",
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "beta": self.config.dpo_beta,
            "max_length": self.config.dataset.max_seq_length,
        }
        import inspect
        sig = inspect.signature(DPOConfig.__init__)
        if "warmup_ratio" in sig.parameters:
            dpo_kwargs["warmup_ratio"] = self.config.optimizer.warmup_ratio
        elif "warmup_steps" in sig.parameters:
            effective_steps = max_steps if (max_steps is not None and max_steps > 0) else 100
            dpo_kwargs["warmup_steps"] = max(0, int(effective_steps * self.config.optimizer.warmup_ratio))
        dpo_config = DPOConfig(**dpo_kwargs)

        peft_config = None
        if self.config.peft is not None and not getattr(self, "is_peft", False):
            from peft import LoraConfig
            peft_cfg = self.config.peft
            peft_config = LoraConfig(
                r=peft_cfg.r,
                lora_alpha=peft_cfg.lora_alpha,
                lora_dropout=peft_cfg.lora_dropout,
                target_modules=peft_cfg.target_modules,
                bias=peft_cfg.bias,
                task_type=peft_cfg.task_type,
                modules_to_save=peft_cfg.modules_to_save,
            )

        return DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=hf_ds,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            callbacks=callbacks or [],
        )

    def train_direct(
        self,
        resume_from: Optional[Any] = None,
        interrupt_after_step: Optional[int] = None,
        cancel_check: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from pathlib import Path
        import itertools
        from llm_training_stack.checkpoints.resume import ResumeManager

        self.setup_model_and_tokenizer()
        self.setup_optimizer()

        start_step = 0
        start_epoch = 0.0
        if resume_from is not None:
            res_info = ResumeManager.resume_into(
                Path(resume_from), self.model, self.optimizer, self.lr_scheduler, ref_model=self.ref_model
            )
            start_step = res_info["step"]
            start_epoch = res_info["epoch"]

        self.model.train()
        beta = self.config.dpo_beta
        global_step = start_step
        grad_accum = self.config.hardware.gradient_accumulation_steps
        optimizer_step = start_step // grad_accum
        max_steps = self.config.max_steps or 20
        loss_history = []
        accumulated_loss = 0.0
        start_time = time.time()
        is_cancelled = False

        batch_iterator = self.get_batches()
        if start_step > 0:
            batch_iterator = itertools.islice(batch_iterator, start_step, None)

        self.optimizer.zero_grad()

        for batch in batch_iterator:
            if (self.output_dir / "cancel.token").exists() or bool(cancel_check and cancel_check()):
                is_cancelled = True
                break
            if global_step >= max_steps:
                break
            if interrupt_after_step is not None and global_step >= interrupt_after_step:
                break

            prompt_len = batch["prompt_len"]
            c_ids = batch["chosen_ids"].to(self.device)
            r_ids = batch["rejected_ids"].to(self.device)

            # Policy model log-probs
            pi_chosen = self._compute_sequence_log_probs(self.model, c_ids, prompt_len)
            pi_rejected = self._compute_sequence_log_probs(self.model, r_ids, prompt_len)

            # Reference model log-probs (no gradients, evaluated per TRL DPOTrainer contract)
            with torch.no_grad():
                if self.is_peft and hasattr(self.model, "disable_adapter"):
                    with self.model.disable_adapter():
                        ref_chosen = self._compute_sequence_log_probs(self.model, c_ids, prompt_len)
                        ref_rejected = self._compute_sequence_log_probs(self.model, r_ids, prompt_len)
                elif self.ref_model is not None:
                    ref_chosen = self._compute_sequence_log_probs(self.ref_model, c_ids, prompt_len)
                    ref_rejected = self._compute_sequence_log_probs(self.ref_model, r_ids, prompt_len)
                else:
                    ref_chosen = torch.zeros_like(pi_chosen)
                    ref_rejected = torch.zeros_like(pi_rejected)

            # Upstream TRL DPO loss formula: -log sigma(beta * ((pi_w - ref_w) - (pi_l - ref_l)))
            pi_logratios = pi_chosen - pi_rejected
            ref_logratios = ref_chosen - ref_rejected
            logits = beta * (pi_logratios - ref_logratios)
            loss = -F.logsigmoid(logits).mean()

            # TRL DPOTrainer implicit reward metrics
            chosen_rewards = (beta * (pi_chosen - ref_chosen)).detach()
            rejected_rewards = (beta * (pi_rejected - ref_rejected)).detach()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

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

                avg_loss = accumulated_loss / grad_accum
                loss_history.append(avg_loss)
                accumulated_loss = 0.0

                self.event_logger.log_event(
                    "dpo_step",
                    global_step + 1,
                    {
                        "loss": round(avg_loss, 5),
                        "margin": round(reward_margin, 4),
                        "reward_accuracy": round(reward_accuracy, 4),
                        "chosen_reward": round(chosen_rewards.mean().item(), 4),
                        "rejected_reward": round(rejected_rewards.mean().item(), 4),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                )

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

        # Flush pending unaccumulated gradients upon dataset exhaustion (if not cancelled)
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
                "task": "dpo",
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

        summary = {
            "status": "COMPLETED",
            "task": "dpo",
            "total_steps": global_step,
            "final_loss": loss_history[-1] if loss_history else None,
            "final_checkpoint": str(final_cp),
        }
        self.manifest["status"] = "COMPLETED"
        self.manifest["final_metrics"] = summary
        RunManifest.save(self.manifest, self.output_dir)
        self.event_logger.close()
        return summary

    def train(
        self,
        resume_from: Optional[Any] = None,
        interrupt_after_step: Optional[int] = None,
        backend: Optional[str] = None,
        cancel_check: Optional[Any] = None,
    ) -> Dict[str, Any]:
        chosen_backend = backend or getattr(self.config, "backend", "transformers")
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
