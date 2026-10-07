"""Supervised Fine-Tuning (SFT) pipeline with strict prompt loss masking."""

from typing import Dict, Iterator, List
import torch
from datasets import load_dataset

from llm_training_stack.pipelines.base import BaseTrainingPipeline, load_strict_dataset


class SFTPipeline(BaseTrainingPipeline):
    """Executes SFT ensuring prompt tokens are masked with labels = -100 and empty targets are rejected."""

    def get_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset_name = self.config.dataset.dataset_name_or_path
        prompt_col = self.config.dataset.prompt_column
        resp_col = self.config.dataset.response_column
        seq_len = self.config.dataset.max_seq_length
        batch_size = self.config.hardware.per_device_train_batch_size

        def mock_sft_factory():
            return [
                {
                    prompt_col: f"Question {i}: Explain LLM gradient accumulation.",
                    resp_col: f"Answer {i}: Gradient accumulation computes gradients over micro-batches before calling optimizer.step().",
                }
                for i in range(100)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_sft_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        current_batch: List[Dict[str, List[int]]] = []

        for idx, item in enumerate(ds):
            if isinstance(item, dict) and prompt_col not in item:
                raise ValueError(
                    f"SFT dataset sample missing prompt column '{prompt_col}'. "
                    f"Available keys: {list(item.keys())}"
                )
            if isinstance(item, dict) and resp_col not in item:
                raise ValueError(
                    f"SFT dataset sample missing response column '{resp_col}'. "
                    f"Available keys: {list(item.keys())}"
                )

            prompt = item[prompt_col] if isinstance(item, dict) else getattr(item, prompt_col, None)
            resp = item[resp_col] if isinstance(item, dict) else getattr(item, resp_col, None)

            # Strict validation: reject empty targets and empty prompts
            if prompt is None or not str(prompt).strip():
                raise ValueError(f"SFT sample at index {idx} contains an empty prompt. Empty prompts are rejected.")
            if resp is None or not str(resp).strip():
                raise ValueError(
                    f"SFT sample at index {idx} contains an empty target response for prompt: "
                    f"'{str(prompt)[:50]}'. Empty targets are rejected."
                )

            # Apply chat template if present, else standard user/assistant formatting
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
                prompt_msgs = [{"role": "user", "content": str(prompt)}]
                full_msgs = [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(resp)},
                ]
                p_ids = self.tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False
                )
                full_ids = self.tokenizer.apply_chat_template(
                    full_msgs, tokenize=True, add_generation_prompt=False, return_dict=False
                )
                if isinstance(p_ids, dict):
                    p_ids = p_ids["input_ids"]
                if isinstance(full_ids, dict):
                    full_ids = full_ids["input_ids"]
            else:
                prompt_text = f"User: {prompt}\nAssistant: "
                resp_text = f"{resp}\n"
                p_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
                r_ids = self.tokenizer.encode(resp_text, add_special_tokens=False)
                full_ids = p_ids + r_ids

            prompt_len = len(p_ids)
            target_token_count = len(full_ids) - prompt_len
            if target_token_count <= 0:
                raise ValueError(
                    f"Empty target tokens: response produced zero tokens after encoding for prompt: '{str(prompt)[:50]}'."
                )

            full_ids = full_ids[:seq_len]
            if len(full_ids) <= prompt_len:
                raise ValueError(
                    f"SFT target response was completely truncated because prompt ({prompt_len} tokens) "
                    f"exceeded or filled max_seq_length ({seq_len}). Increase max_seq_length."
                )

            # Strict prompt loss masking: labels = -100 on all prompt tokens
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            current_batch.append({"input_ids": full_ids, "labels": labels})

            if len(current_batch) == batch_size:
                # Pad batch dynamically
                max_len = max(len(x["input_ids"]) for x in current_batch)
                pad_id = self.tokenizer.pad_token_id or 0

                batch_input_ids = []
                batch_attn_masks = []
                batch_labels = []

                for item in current_batch:
                    pad_len = max_len - len(item["input_ids"])
                    b_ids = item["input_ids"] + [pad_id] * pad_len
                    b_mask = [1] * len(item["input_ids"]) + [0] * pad_len
                    b_lbl = item["labels"] + [-100] * pad_len

                    batch_input_ids.append(b_ids)
                    batch_attn_masks.append(b_mask)
                    batch_labels.append(b_lbl)

                current_batch = []
                yield {
                    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_attn_masks, dtype=torch.long),
                    "labels": torch.tensor(batch_labels, dtype=torch.long),
                }

        # Tail batch handling: pad and yield any remaining samples so len(ds) % batch_size != 0 items are not dropped
        if current_batch:
            max_len = max(len(x["input_ids"]) for x in current_batch)
            pad_id = self.tokenizer.pad_token_id or 0

            batch_input_ids = []
            batch_attn_masks = []
            batch_labels = []

            for item in current_batch:
                pad_len = max_len - len(item["input_ids"])
                b_ids = item["input_ids"] + [pad_id] * pad_len
                b_mask = [1] * len(item["input_ids"]) + [0] * pad_len
                b_lbl = item["labels"] + [-100] * pad_len

                batch_input_ids.append(b_ids)
                batch_attn_masks.append(b_mask)
                batch_labels.append(b_lbl)

            yield {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attn_masks, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

    def build_trainer(self, callbacks=None, interrupt_after_step=None):
        """Constructs upstream trl.SFTTrainer for supervised fine-tuning."""
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig

        dataset_name = self.config.dataset.dataset_name_or_path
        prompt_col = self.config.dataset.prompt_column
        resp_col = self.config.dataset.response_column

        def mock_sft_factory():
            return [
                {
                    prompt_col: f"Question {i}: Explain LLM gradient accumulation.",
                    resp_col: f"Answer {i}: Gradient accumulation computes gradients over micro-batches before calling optimizer.step().",
                }
                for i in range(100)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_sft_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        formatted_samples = []
        for idx, item in enumerate(ds):
            if isinstance(item, dict) and prompt_col not in item:
                raise ValueError(f"SFT dataset sample missing prompt column '{prompt_col}'.")
            if isinstance(item, dict) and resp_col not in item:
                raise ValueError(f"SFT dataset sample missing response column '{resp_col}'.")

            prompt = item[prompt_col] if isinstance(item, dict) else getattr(item, prompt_col, None)
            resp = item[resp_col] if isinstance(item, dict) else getattr(item, resp_col, None)

            if prompt is None or not str(prompt).strip():
                raise ValueError(f"SFT sample at index {idx} contains an empty prompt. Empty prompts are rejected.")
            if resp is None or not str(resp).strip():
                raise ValueError(f"SFT sample at index {idx} contains an empty target response. Empty targets are rejected.")

            formatted_samples.append({
                "prompt": str(prompt),
                "completion": str(resp),
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

        sft_kwargs = {
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
            "max_length": self.config.dataset.max_seq_length,
        }
        import inspect
        sig = inspect.signature(SFTConfig.__init__)
        if "warmup_ratio" in sig.parameters:
            sft_kwargs["warmup_ratio"] = self.config.optimizer.warmup_ratio
        elif "warmup_steps" in sig.parameters:
            effective_steps = max_steps if (max_steps is not None and max_steps > 0) else 100
            sft_kwargs["warmup_steps"] = max(0, int(effective_steps * self.config.optimizer.warmup_ratio))
        sft_config = SFTConfig(**sft_kwargs)

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

        return SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=hf_ds,
            processing_class=self.tokenizer,
            peft_config=peft_config,
            callbacks=callbacks or [],
        )
