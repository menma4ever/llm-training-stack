"""Pydantic v2 configuration schemas for LLM Training Stack."""

from enum import Enum
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator, field_validator


class TaskType(str, Enum):
    CPT = "cpt"    # Continued Pre-Training (auto-regressive raw text)
    SFT = "sft"    # Supervised Fine-Tuning (instruction / chat with loss masking)
    LORA = "lora"  # Parameter-Efficient Fine-Tuning via LoRA
    DPO = "dpo"    # Direct Preference Optimization (pairwise alignment)


class ModelConfig(BaseModel):
    model_name_or_path: str = Field(..., description="Hugging Face model id or local checkpoint path")
    tokenizer_name_or_path: Optional[str] = Field(None, description="Tokenizer path if different from model")
    trust_remote_code: bool = Field(False, description="Allow custom model code execution")
    torch_dtype: Literal["float32", "bfloat16", "float16", "auto"] = Field("auto", description="Model tensor dtype")
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = Field(
        "sdpa", description="Attention kernel implementation"
    )
    model_max_length: Optional[int] = Field(None, description="Override tokenizer max length")
    revision: Optional[str] = Field(None, description="Hugging Face repo commit hash or branch tag")
    model_revision: Optional[str] = Field(None, description="Alias for revision")

    @model_validator(mode="after")
    def sync_model_revisions(self) -> "ModelConfig":
        if self.revision is not None and self.model_revision is not None:
            if self.revision != self.model_revision:
                raise ValueError(
                    f"Conflicting revisions specified in ModelConfig: revision='{self.revision}' "
                    f"and model_revision='{self.model_revision}'"
                )
        elif self.revision is not None:
            self.model_revision = self.revision
        elif self.model_revision is not None:
            self.revision = self.model_revision
        return self


class DatasetConfig(BaseModel):
    dataset_name_or_path: str = Field(..., description="Dataset name on HF Hub or path to local json/jsonl/parquet")
    train_split: str = Field("train", description="Dataset split for training")
    eval_split: Optional[str] = Field("validation", description="Dataset split for evaluation")
    text_column: str = Field("text", description="Primary text column for CPT")
    prompt_column: str = Field("prompt", description="Prompt column for SFT/DPO")
    response_column: str = Field("response", description="Response column for SFT")
    chosen_column: str = Field("chosen", description="Winning response column for DPO")
    rejected_column: str = Field("rejected", description="Losing response column for DPO")
    max_seq_length: int = Field(2048, ge=32, le=131072, description="Maximum sequence length in tokens")
    packing: bool = Field(False, description="Pack multiple short examples into max_seq_length sequences")
    train_sample_limit: Optional[int] = Field(None, ge=1, description="Truncate dataset to N samples for testing/smoke runs")


class PeftConfig(BaseModel):
    r: int = Field(16, ge=1, le=512, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha scaling parameter")
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Linear module names to attach LoRA adapters to"
    )
    bias: Literal["none", "all", "lora_only"] = Field("none", description="LoRA bias training policy")
    task_type: str = Field("CAUSAL_LM", description="PEFT task type")
    modules_to_save: Optional[List[str]] = Field(None, description="Additional modules to unfreeze and train")


class OptimizerConfig(BaseModel):
    optimizer_type: Literal["adamw_torch", "adamw_hf", "sgd", "adamw_torch_fused"] = Field(
        "adamw_torch", description="Optimizer implementation"
    )
    learning_rate: float = Field(2e-5, gt=0.0, description="Peak learning rate")
    weight_decay: float = Field(0.01, ge=0.0, description="L2 weight decay")
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.5, description="Proportion of training steps for linear warmup")
    lr_scheduler_type: Literal["cosine", "linear", "constant", "constant_with_warmup"] = Field(
        "cosine", description="Learning rate scheduler"
    )
    adam_beta1: float = Field(0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(0.999, ge=0.0, lt=1.0)
    adam_epsilon: float = Field(1e-8, gt=0.0)
    max_grad_norm: float = Field(1.0, ge=0.0, description="Gradient clipping maximum norm")


class HardwareConfig(BaseModel):
    target_device: Literal["auto", "cpu", "cuda"] = Field("auto", description="Compute target device")
    per_device_train_batch_size: int = Field(2, ge=1, description="Micro-batch size per device for training")
    per_device_eval_batch_size: int = Field(2, ge=1, description="Micro-batch size per device for evaluation")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(False, description="Enable activation checkpointing to save memory")
    mixed_precision: Literal["no", "fp16", "bf16"] = Field("no", description="Mixed precision mode")
    dataloader_num_workers: int = Field(0, ge=0, description="Number of background dataloader processes")


class LoggingConfig(BaseModel):
    output_dir: str = Field("./runs", description="Directory to persist checkpoints, events and manifests")
    run_name: Optional[str] = Field(None, description="Custom name for the training run")
    logging_steps: int = Field(10, ge=1, description="Frequency of metric logging in steps")
    eval_steps: int = Field(50, ge=1, description="Frequency of evaluation passes in steps")
    save_steps: int = Field(100, ge=1, description="Frequency of checkpoint saves in steps")
    save_total_limit: int = Field(3, ge=1, description="Maximum number of older checkpoints to retain")


class TrainingJobConfig(BaseModel):
    schema_version: str = Field("1.0.0", description="Configuration schema version")
    task_type: TaskType = Field(..., description="Training task type (cpt, sft, lora, dpo)")
    model: ModelConfig = Field(..., description="Base model specifications")
    dataset: DatasetConfig = Field(..., description="Dataset input and column mappings")
    peft: Optional[PeftConfig] = Field(None, description="PEFT/LoRA configuration")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer & scheduler settings")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="Hardware & batch sizing settings")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Output paths and telemetry frequency")
    epochs: int = Field(1, ge=1, description="Number of complete training epochs")
    max_steps: Optional[int] = Field(None, ge=1, description="Maximum total training steps override")
    seed: int = Field(42, description="Global random seed for deterministic reproducibility")
    dpo_beta: float = Field(0.1, gt=0.0, le=1.0, description="DPO temperature scale parameter")

    adaptation_mode: Optional[Literal["full", "lora"]] = Field(
        None, description="Adaptation mode: 'full' parameter tuning or 'lora' adapter tuning"
    )
    backend: Literal["transformers", "direct"] = Field(
        "transformers", description="Execution backend engine: 'transformers' (default) or 'direct' (PyTorch loop)"
    )
    revision: Optional[str] = Field(None, description="Job-level shortcut or alias for model.revision")
    model_revision: Optional[str] = Field(None, description="Job-level shortcut or alias for model.revision")

    @property
    def effective_batch_size(self) -> int:
        return self.hardware.per_device_train_batch_size * self.hardware.gradient_accumulation_steps

    @model_validator(mode="after")
    def validate_task_consistency(self) -> "TrainingJobConfig":
        # Synchronize revision / model_revision across Job and ModelConfig without collision
        job_rev = self.revision or self.model_revision
        if self.revision is not None and self.model_revision is not None:
            if self.revision != self.model_revision:
                raise ValueError(
                    f"Conflicting job-level revisions specified: revision='{self.revision}' "
                    f"and model_revision='{self.model_revision}'"
                )

        if job_rev is not None:
            if self.model.revision is not None and self.model.revision != job_rev:
                raise ValueError(
                    f"Conflicting revisions between job ({job_rev}) and model ({self.model.revision})."
                )
            self.model.revision = job_rev
            self.model.model_revision = job_rev
            self.revision = job_rev
            self.model_revision = job_rev
        elif self.model.revision is not None:
            self.revision = self.model.revision
            self.model_revision = self.model.revision

        if self.task_type == TaskType.LORA:
            if self.peft is None:
                self.peft = PeftConfig()
            if self.adaptation_mode is None:
                self.adaptation_mode = "lora"

        if self.adaptation_mode == "lora" and self.peft is None:
            self.peft = PeftConfig()
        elif self.peft is not None and self.adaptation_mode is None:
            self.adaptation_mode = "lora"
        elif self.peft is None and self.adaptation_mode is None:
            self.adaptation_mode = "full"

        if self.task_type == TaskType.DPO and not (self.dataset.chosen_column and self.dataset.rejected_column):
            raise ValueError(
                "DPO task requires both 'chosen_column' and 'rejected_column' to be configured in dataset."
            )

        return self
