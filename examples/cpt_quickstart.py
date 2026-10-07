"""Quickstart example: Continued Pre-Training (CPT)."""

from pathlib import Path
from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.pipelines.cpt import CPTPipeline

config = TrainingJobConfig(
    task_type=TaskType.CPT,
    model=ModelConfig(
        model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype="float32",
    ),
    dataset=DatasetConfig(
        dataset_name_or_path="synthetic",
        text_column="text",
        max_seq_length=256,
        train_sample_limit=20,
    ),
    hardware=HardwareConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        target_device="cpu",
    ),
    logging=LoggingConfig(
        output_dir="./runs/cpt_quickstart",
        save_steps=5,
    ),
    max_steps=6,
)

pipeline = CPTPipeline(config)
result = pipeline.train()
print("CPT Training Complete:", result)
