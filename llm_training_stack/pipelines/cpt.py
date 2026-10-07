"""Continued Pre-Training (CPT) pipeline for raw text domain adaptation."""

from typing import Dict, Iterator, List
import torch
from datasets import load_dataset

from llm_training_stack.pipelines.base import BaseTrainingPipeline, load_strict_dataset


class CPTPipeline(BaseTrainingPipeline):
    """Executes Continued Pre-Training with causal language modeling objectives."""

    def get_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset_name = self.config.dataset.dataset_name_or_path
        text_col = self.config.dataset.text_column
        seq_len = self.config.dataset.max_seq_length
        batch_size = self.config.hardware.per_device_train_batch_size

        def mock_cpt_factory():
            return [
                {"text": f"Machine learning training sample {i} for continued pretraining domain adaptation."}
                for i in range(100)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_cpt_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        max_steps = self.config.max_steps or 100
        steps_yielded = 0
        buffer_ids: List[int] = []

        for epoch in range(max(1, self.config.epochs * 10)):
            for item in ds:
                if isinstance(item, dict) and text_col not in item:
                    raise ValueError(
                        f"CPT dataset item missing required text column '{text_col}'. "
                        f"Available keys: {list(item.keys())}"
                    )
                text = item[text_col] if isinstance(item, dict) else getattr(item, text_col, "")
                if not text or not str(text).strip():
                    continue
                tokenized = self.tokenizer.encode(text, add_special_tokens=True)
                buffer_ids.extend(tokenized)

                while len(buffer_ids) >= seq_len * batch_size:
                    batch_chunks = []
                    for b in range(batch_size):
                        chunk = buffer_ids[:seq_len]
                        buffer_ids = buffer_ids[seq_len:]
                        batch_chunks.append(chunk)

                    input_ids = torch.tensor(batch_chunks, dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    labels = input_ids.clone()

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                    steps_yielded += 1
                    if steps_yielded >= max_steps:
                        return

    def build_trainer(self, callbacks=None, interrupt_after_step=None):
        """Constructs upstream transformers.Trainer for CPT causal language modeling."""
        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, Trainer

        dataset_name = self.config.dataset.dataset_name_or_path
        text_col = self.config.dataset.text_column
        seq_len = self.config.dataset.max_seq_length

        def mock_cpt_factory():
            return [
                {"text": f"Machine learning training sample {i} for continued pretraining domain adaptation."}
                for i in range(100)
            ]

        ds = load_strict_dataset(
            dataset_name,
            split=self.config.dataset.train_split,
            mock_factory=mock_cpt_factory,
        )

        if self.config.dataset.train_sample_limit:
            ds = ds[:self.config.dataset.train_sample_limit]

        raw_texts = []
        for item in ds:
            if isinstance(item, dict) and text_col not in item:
                raise ValueError(
                    f"CPT dataset item missing required text column '{text_col}'. "
                    f"Available keys: {list(item.keys())}"
                )
            text = item[text_col] if isinstance(item, dict) else getattr(item, text_col, "")
            if text and str(text).strip():
                raw_texts.append(str(text))

        hf_ds = Dataset.from_dict({"text": raw_texts})

        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=seq_len)

        tokenized_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        args = self.build_training_arguments(interrupt_after_step=interrupt_after_step)

        return Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_ds,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks or [],
        )
