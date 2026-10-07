"""Integration tests for CPT, SFT with prompt masking, LoRA, and DPO pipelines."""

import json
import pytest
import torch
from pathlib import Path

from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    PeftConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.pipelines.cpt import CPTPipeline
from llm_training_stack.pipelines.sft import SFTPipeline
from llm_training_stack.pipelines.lora import LoRAPipeline
from llm_training_stack.pipelines.dpo import DPOPipeline




def test_sft_prompt_loss_masking_correctness(base_test_config):
    """Verifies that SFT prompt tokens receive label = -100 (never contributing to loss)."""
    pipeline = SFTPipeline(base_test_config)
    pipeline.setup_model_and_tokenizer()

    batches = list(pipeline.get_batches())
    assert len(batches) > 0

    first_batch = batches[0]
    input_ids = first_batch["input_ids"]
    labels = first_batch["labels"]

    # Verify -100 appears in labels (prompt tokens masked)
    has_masked_tokens = (labels == -100).any().item()
    assert has_masked_tokens is True, "Prompt tokens were not masked with -100 in SFT labels!"

    # Verify assistant response tokens have valid non-negative labels
    has_unmasked_tokens = (labels >= 0).any().item()
    assert has_unmasked_tokens is True, "No assistant tokens remained unmasked for gradient updates!"


def test_lora_pipeline_execution(tmp_path, base_test_config):
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.LORA
    cfg.logging.output_dir = str(tmp_path / "lora_run")
    cfg.peft = PeftConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])

    pipeline = LoRAPipeline(cfg)
    summary = pipeline.train()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 4
    assert summary["final_loss"] is not None

    stats = pipeline.manifest["peft_stats"]
    assert stats["trainable_percent"] < 10.0, "LoRA trainable parameter count should be <10% of total!"
    assert (Path(summary["final_checkpoint"]) / "adapter_config.json").exists()


def test_cpt_pipeline_execution(tmp_path, base_test_config):
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.CPT
    cfg.logging.output_dir = str(tmp_path / "cpt_run")

    pipeline = CPTPipeline(cfg)
    summary = pipeline.train()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 4
    assert summary["final_loss"] is not None


def test_dpo_pipeline_execution(tmp_path, base_test_config):
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.DPO
    cfg.logging.output_dir = str(tmp_path / "dpo_run")
    cfg.dpo_beta = 0.1
    cfg.max_steps = 2

    pipeline = DPOPipeline(cfg)
    summary = pipeline.train()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 2
    assert summary["final_loss"] is not None


def test_strict_dataset_loading_rejection(tmp_path, base_test_config):
    """Verifies that pipelines raise RuntimeError on non-existent datasets instead of silently mocking."""
    cfg = base_test_config.model_copy(deep=True)
    cfg.dataset.dataset_name_or_path = "nonexistent_real_dataset_path_12345"

    cpt_pipe = CPTPipeline(cfg)
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        list(cpt_pipe.get_batches())

    sft_pipe = SFTPipeline(cfg)
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        list(sft_pipe.get_batches())

    dpo_pipe = DPOPipeline(cfg)
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        list(dpo_pipe.get_batches())


def test_sft_empty_target_rejection(tmp_path, base_test_config):
    """Verifies that SFTPipeline strictly rejects empty target responses."""
    bad_data = [
        {"prompt": "Hello there", "response": ""},
        {"prompt": "Another prompt", "response": "   \n"},
    ]
    data_file = tmp_path / "bad_sft.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for item in bad_data:
            f.write(json.dumps(item) + "\n")

    cfg = base_test_config.model_copy(deep=True)
    cfg.dataset.dataset_name_or_path = str(data_file)

    pipeline = SFTPipeline(cfg)
    pipeline.setup_model_and_tokenizer()

    with pytest.raises(ValueError, match="Empty targets are rejected"):
        list(pipeline.get_batches())


def test_sft_empty_prompt_rejection(tmp_path, base_test_config):
    """Verifies that SFTPipeline strictly rejects empty prompts."""
    bad_data = [
        {"prompt": "", "response": "Valid response."},
    ]
    data_file = tmp_path / "bad_prompt_sft.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for item in bad_data:
            f.write(json.dumps(item) + "\n")

    cfg = base_test_config.model_copy(deep=True)
    cfg.dataset.dataset_name_or_path = str(data_file)

    pipeline = SFTPipeline(cfg)
    pipeline.setup_model_and_tokenizer()

    with pytest.raises(ValueError, match="Empty prompts are rejected"):
        list(pipeline.get_batches())


def test_dpo_empty_target_and_identical_pair_rejection(tmp_path, base_test_config):
    """Verifies that DPOPipeline strictly rejects empty chosen/rejected responses and identical pairs."""
    # 1. Empty chosen response
    bad_data_1 = [{"prompt": "P1", "chosen": "", "rejected": "R1"}]
    f1 = tmp_path / "dpo_empty_chosen.jsonl"
    with open(f1, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_data_1[0]) + "\n")

    cfg1 = base_test_config.model_copy(deep=True)
    cfg1.task_type = TaskType.DPO
    cfg1.dataset.dataset_name_or_path = str(f1)
    pipe1 = DPOPipeline(cfg1)
    pipe1.setup_model_and_tokenizer()
    with pytest.raises(ValueError, match="empty chosen response"):
        list(pipe1.get_batches())

    # 2. Identical chosen and rejected responses
    bad_data_2 = [{"prompt": "P1", "chosen": "Same text", "rejected": "Same text"}]
    f2 = tmp_path / "dpo_identical.jsonl"
    with open(f2, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_data_2[0]) + "\n")

    cfg2 = base_test_config.model_copy(deep=True)
    cfg2.task_type = TaskType.DPO
    cfg2.dataset.dataset_name_or_path = str(f2)
    pipe2 = DPOPipeline(cfg2)
    pipe2.setup_model_and_tokenizer()
    with pytest.raises(ValueError, match="identical chosen and rejected"):
        list(pipe2.get_batches())


def test_cpt_orthogonal_lora_adaptation(tmp_path, base_test_config):
    """Verifies that CPT pipeline can be run with LoRA adapters (orthogonal adaptation mode)."""
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.CPT
    cfg.adaptation_mode = "lora"
    cfg.peft = PeftConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    cfg.logging.output_dir = str(tmp_path / "cpt_lora_run")
    cfg.max_steps = 2

    pipeline = CPTPipeline(cfg)
    summary = pipeline.train()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 2
    assert pipeline.is_peft is True
    assert (Path(summary["final_checkpoint"]) / "adapter_config.json").exists()


def test_dpo_orthogonal_lora_adaptation(tmp_path, base_test_config):
    """Verifies that DPO pipeline can be run with LoRA adapters (orthogonal adaptation mode)."""
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.DPO
    cfg.adaptation_mode = "lora"
    cfg.peft = PeftConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    cfg.logging.output_dir = str(tmp_path / "dpo_lora_run")
    cfg.max_steps = 2

    pipeline = DPOPipeline(cfg)
    summary = pipeline.train()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 2
    assert pipeline.is_peft is True
    assert (Path(summary["final_checkpoint"]) / "adapter_config.json").exists()


def test_smollm_135m_open_weight_smoke_run(tmp_path):
    """Verifies real open-weight pretrained causal language model (SmolLM-135M)

    adapts over real instruction data via SFTPipeline with valid loss and checkpoint.
    """
    import json
    data_file = tmp_path / "instructions.jsonl"
    data = [
        {"prompt": "Define gradient accumulation.", "response": "Accumulating gradients over micro-batches before stepping optimizer."},
        {"prompt": "What is LoRA?", "response": "Low-Rank Adaptation injects trainable rank decomposition matrices into layers."},
        {"prompt": "What is DPO?", "response": "Direct Preference Optimization directly aligns language models from preference pairs."},
        {"prompt": "Why mask prompt tokens?", "response": "Masking prompt tokens ensures loss is only computed on generating the response."},
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(
            model_name_or_path="HuggingFaceTB/SmolLM-135M",
            tokenizer_name_or_path="HuggingFaceTB/SmolLM-135M",
            revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
            torch_dtype="float32",
        ),
        dataset=DatasetConfig(
            dataset_name_or_path=str(data_file),
            max_seq_length=32,
            train_sample_limit=4,
        ),
        hardware=HardwareConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            target_device="cpu",
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "smollm_smoke"),
            save_steps=2,
        ),
        max_steps=2,
    )

    pipe = SFTPipeline(cfg)
    summary = pipe.train()
    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 2
    assert summary["final_loss"] is not None
    assert (tmp_path / "smollm_smoke" / "manifest.json").exists()
    assert (tmp_path / "smollm_smoke" / "checkpoint-2").exists()


def test_sft_tail_batch_preservation(tmp_path, base_test_config):
    """Verifies that SFTPipeline yields tail batch when len(ds) % batch_size != 0."""
    import json
    data_file = tmp_path / "odd_sft.jsonl"
    odd_data = [
        {"prompt": f"Question {i}", "response": f"Detailed answer number {i} for evaluation."}
        for i in range(5)
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for item in odd_data:
            f.write(json.dumps(item) + "\n")

    cfg = base_test_config.model_copy(deep=True)
    cfg.dataset.dataset_name_or_path = str(data_file)
    cfg.dataset.train_sample_limit = 5
    cfg.hardware.per_device_train_batch_size = 2

    pipe = SFTPipeline(cfg)
    pipe.setup_model_and_tokenizer()

    batches = list(pipe.get_batches())
    # 5 samples with batch_size=2 should yield 3 batches: batch 1 (2), batch 2 (2), batch 3 (1)
    assert len(batches) == 3
    assert batches[0]["input_ids"].size(0) == 2
    assert batches[1]["input_ids"].size(0) == 2
    assert batches[2]["input_ids"].size(0) == 1


def test_dpo_direct_cancel_check(tmp_path, base_test_config):
    """Verifies that DPO direct loop halts upon cancel_check returning True, saving checkpoint and CANCELLED status."""
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.DPO
    cfg.backend = "direct"
    cfg.logging.output_dir = str(tmp_path / "dpo_cancel_direct")
    cfg.dpo_beta = 0.1
    cfg.max_steps = 10

    counter = {"calls": 0}
    def check_fn():
        counter["calls"] += 1
        return counter["calls"] > 2

    pipeline = DPOPipeline(cfg)
    summary = pipeline.train(cancel_check=check_fn)

    assert summary["status"] == "CANCELLED"
    assert summary["total_steps"] <= 3
    assert Path(summary["final_checkpoint"]).exists()
    assert (tmp_path / "dpo_cancel_direct" / "manifest.json").exists()
    with open(tmp_path / "dpo_cancel_direct" / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "CANCELLED"


def test_dpo_upstream_cancel_check(tmp_path, base_test_config):
    """Verifies that DPO upstream trainer halts upon cancel_check returning True, saving checkpoint and CANCELLED status."""
    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.DPO
    cfg.backend = "transformers"
    cfg.logging.output_dir = str(tmp_path / "dpo_cancel_upstream")
    cfg.dpo_beta = 0.1
    cfg.max_steps = 10

    counter = {"calls": 0}
    def check_fn():
        counter["calls"] += 1
        return counter["calls"] > 2

    pipeline = DPOPipeline(cfg)
    summary = pipeline.train(cancel_check=check_fn)

    assert summary["status"] == "CANCELLED"
    assert summary["total_steps"] < cfg.max_steps, f"Expected halted before max_steps={cfg.max_steps}, got {summary['total_steps']}"
    assert summary["total_steps"] <= 3, f"Expected stopped promptly upon cancellation, got step {summary['total_steps']}"
    assert Path(summary["final_checkpoint"]).exists()
    with open(tmp_path / "dpo_cancel_upstream" / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["status"] == "CANCELLED"
    assert manifest["final_metrics"]["total_steps"] < cfg.max_steps



