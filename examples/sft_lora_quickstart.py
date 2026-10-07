"""Quickstart example: Supervised Fine-Tuning with LoRA."""

from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    PeftConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.pipelines.lora import LoRAPipeline

config = TrainingJobConfig(
    task_type=TaskType.LORA,
    model=ModelConfig(
        model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype="float32",
    ),
    dataset=DatasetConfig(
        dataset_name_or_path="synthetic",
        prompt_column="prompt",
        response_column="response",
        max_seq_length=256,
        train_sample_limit=30,
    ),
    peft=PeftConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
    ),
    hardware=HardwareConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        target_device="cpu",
    ),
    logging=LoggingConfig(
        output_dir="./runs/sft_lora_quickstart",
        save_steps=5,
    ),
    max_steps=6,
)

pipeline = LoRAPipeline(config)
result = pipeline.train()
print("LoRA Fine-Tuning Complete:", result)
