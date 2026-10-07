"""Quickstart example: Direct Preference Optimization (DPO)."""

from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.pipelines.dpo import DPOPipeline

config = TrainingJobConfig(
    task_type=TaskType.DPO,
    model=ModelConfig(
        model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype="float32",
    ),
    dataset=DatasetConfig(
        dataset_name_or_path="synthetic",
        prompt_column="prompt",
        chosen_column="chosen",
        rejected_column="rejected",
        max_seq_length=256,
        train_sample_limit=20,
    ),
    hardware=HardwareConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        target_device="cpu",
    ),
    logging=LoggingConfig(
        output_dir="./runs/dpo_quickstart",
        save_steps=4,
    ),
    dpo_beta=0.1,
    max_steps=4,
)

pipeline = DPOPipeline(config)
result = pipeline.train()
print("DPO Alignment Complete:", result)
