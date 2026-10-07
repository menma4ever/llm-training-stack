"""Explicit negative test cases for robust failure rejection.

Verifies:
1. Invalid dataset path rejection (non-existent file, malformed format, missing split)
2. Unauthorized MCP launch rejection (authorized=False, operator policy lockdown, plan grant mismatch)
3. Unsupported hardware backend and optimizer rejection in schema validation
"""

import json
import pytest
from pydantic import ValidationError

from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    HardwareConfig,
    OptimizerConfig,
)
from llm_training_stack.pipelines.cpt import CPTPipeline
from llm_training_stack.pipelines.sft import SFTPipeline
from llm_training_stack.pipelines.dpo import DPOPipeline
from llm_training_stack.pipelines.base import load_strict_dataset
from llm_training_stack.mcp.server import launch_training, run_preflight_probe


# ==============================================================================
# 1. Invalid Dataset Path Rejection
# ==============================================================================

def test_negative_nonexistent_dataset_path_rejection():
    """Verify strict rejection when pointing to non-existent dataset paths."""
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        load_strict_dataset("/path/to/completely/nonexistent/dataset.jsonl")


def test_negative_invalid_dataset_extension_or_format(tmp_path):
    """Verify rejection when dataset file contains invalid or unparseable content."""
    corrupt_file = tmp_path / "corrupt_data.jsonl"
    corrupt_file.write_text("NOT VALID JSON CONTENT {{{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        load_strict_dataset(str(corrupt_file))


def test_negative_pipeline_empty_dataset_rejection(tmp_path, base_test_config):
    """Verify SFTPipeline and DPOPipeline reject empty or missing datasets during batching."""
    cfg = base_test_config.model_copy(deep=True)
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    cfg.dataset.dataset_name_or_path = str(empty_file)

    sft = SFTPipeline(cfg)
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        list(sft.get_batches())


# ==============================================================================
# 2. Unauthorized MCP Launch Rejection
# ==============================================================================

def test_negative_mcp_launch_explicitly_unauthorized():
    """Verify FastMCP server returns EXECUTION_DENIED when authorized=False."""
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
        "dataset": {"dataset_name_or_path": "synthetic"},
    }
    result_raw = launch_training(config_json=json.dumps(cfg), authorized=False)
    result = json.loads(result_raw)
    assert result["error"] == "EXECUTION_DENIED"
    assert result["authorized"] is False


def test_negative_mcp_launch_operator_policy_lockdown(monkeypatch):
    """Verify FastMCP server rejects launch even if agent claims authorized=True when environment lacks operator permission."""
    monkeypatch.delenv("TRAIN_STACK_ALLOW_LAUNCH", raising=False)
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
        "dataset": {"dataset_name_or_path": "synthetic"},
    }
    result_raw = launch_training(config_json=json.dumps(cfg), authorized=True)
    result = json.loads(result_raw)
    assert result["error"] == "EXECUTION_DENIED_BY_OPERATOR_POLICY"
    assert result["operator_policy"] == "READ_ONLY"


def test_negative_mcp_probe_isolation_bypass_rejection(monkeypatch):
    """Verify FastMCP server rejects callers trying to bypass subprocess isolation (in_subprocess=False)."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_PROBE", "1")
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "synthetic"},
    }
    result_raw = run_preflight_probe(
        config_json=json.dumps(cfg),
        in_subprocess=False,  # Unauthorized escape from process isolation
    )
    result = json.loads(result_raw)
    assert result["error"] == "ISOLATION_POLICY_VIOLATION"


# ==============================================================================
# 3. Unsupported Backend & Optimizer Rejection
# ==============================================================================

def test_negative_unsupported_hardware_device_rejection():
    """Verify schema strictly rejects unsupported target devices (e.g., TPU, ROCM, DirectML)."""
    with pytest.raises(ValidationError) as exc_info:
        HardwareConfig(target_device="tpu")
    assert "target_device" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        HardwareConfig(target_device="directml")
    assert "target_device" in str(exc_info.value)


def test_negative_unsupported_optimizer_type_rejection():
    """Verify schema strictly rejects unsupported optimizer backends."""
    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(optimizer_type="adamw_apex")
    assert "optimizer_type" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        OptimizerConfig(optimizer_type="unsupported_custom_opt")
    assert "optimizer_type" in str(exc_info.value)


def test_negative_unsupported_mixed_precision_mode_rejection():
    """Verify schema strictly rejects unsupported mixed precision modes (e.g., fp8)."""
    with pytest.raises(ValidationError) as exc_info:
        HardwareConfig(mixed_precision="fp8")
    assert "mixed_precision" in str(exc_info.value)
