"""Tests for configuration schemas, validation rules, and file loaders."""

import pytest
from pydantic import ValidationError
from llm_training_stack.config.schema import (
    TaskType,
    TrainingJobConfig,
    ModelConfig,
    DatasetConfig,
    PeftConfig,
    OptimizerConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.config.loader import ConfigLoader


def test_valid_sft_config():
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset"),
    )
    assert cfg.task_type == TaskType.SFT
    assert cfg.effective_batch_size == 8  # default: 2 * 4
    assert cfg.optimizer.learning_rate == 2e-5


def test_lora_auto_initialization():
    cfg = TrainingJobConfig(
        task_type=TaskType.LORA,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset"),
    )
    assert cfg.peft is not None
    assert cfg.peft.r == 16
    assert "q_proj" in cfg.peft.target_modules


def test_dpo_validation_requires_columns():
    with pytest.raises(ValidationError):
        TrainingJobConfig(
            task_type=TaskType.DPO,
            model=ModelConfig(model_name_or_path="test/model"),
            dataset=DatasetConfig(
                dataset_name_or_path="test/dataset",
                chosen_column="",  # Missing required chosen column
                rejected_column="",
            ),
        )


def test_yaml_json_loader_roundtrip(tmp_path):
    cfg = TrainingJobConfig(
        task_type=TaskType.CPT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset", max_seq_length=1024),
    )
    yaml_file = tmp_path / "test_config.yaml"
    ConfigLoader.save_to_file(cfg, yaml_file)
    loaded_yaml = ConfigLoader.load_from_file(yaml_file)
    assert loaded_yaml.task_type == TaskType.CPT
    assert loaded_yaml.dataset.max_seq_length == 1024

    json_file = tmp_path / "test_config.json"
    ConfigLoader.save_to_file(cfg, json_file)
    loaded_json = ConfigLoader.load_from_file(json_file)
    assert loaded_json.task_type == TaskType.CPT
