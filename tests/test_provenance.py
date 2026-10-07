"""Tests for run manifests and structured event logging."""

import json
from pathlib import Path
from llm_training_stack.config.schema import TrainingJobConfig, TaskType, ModelConfig, DatasetConfig
from llm_training_stack.provenance.manifest import RunManifest
from llm_training_stack.provenance.event_logger import StructuredEventLogger


def test_manifest_creation_and_integrity(tmp_path):
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset"),
    )
    manifest = RunManifest.create(cfg)
    assert manifest["status"] == "INITIALIZED"
    assert "torch" in manifest["libraries"]
    assert "transformers" in manifest["libraries"]

    saved_path = RunManifest.save(manifest, tmp_path)
    assert saved_path.exists()

    loaded = RunManifest.load(saved_path)
    assert loaded["run_id"] == manifest["run_id"]
    assert loaded["config"]["task_type"] == "sft"


def test_structured_event_logger(tmp_path):
    log_file = tmp_path / "events.jsonl"
    with StructuredEventLogger(log_file) as logger:
        logger.log_step(
            step=1,
            loss=2.5432,
            lr=1e-5,
            epoch=0.1,
            step_time_ms=120.5,
            tokens_per_sec=1500.0,
            grad_norm=0.45,
            memory_mb=512.0,
        )

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "step_metric"
    assert event["step"] == 1
    assert event["loss"] == 2.5432


def test_immutable_launch_manifest_contract(tmp_path):
    """Verifies that launch manifest cannot be overwritten post-launch and validates tamper detection."""
    import pytest
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset"),
    )
    manifest = RunManifest.create(cfg)
    assert "launch_digest_sha256" in manifest

    launch_path = RunManifest.save_launch_manifest(manifest, tmp_path)
    assert launch_path.exists()
    assert RunManifest.verify_launch_manifest(tmp_path) is True

    # Attempting to overwrite launch_manifest raises RuntimeError
    with pytest.raises(RuntimeError, match="Launch manifest is immutable"):
        RunManifest.save_launch_manifest(manifest, tmp_path)

    # Tampering with launch_manifest invalidates verification
    with open(launch_path, "r", encoding="utf-8") as f:
        tampered_data = json.load(f)
    tampered_data["config"]["task_type"] = "tampered_type"
    with open(launch_path, "w", encoding="utf-8") as f:
        json.dump(tampered_data, f)

    assert RunManifest.verify_launch_manifest(tmp_path) is False


def test_installed_package_provenance_site_packages():
    """Verifies that when running in installed candidate mode, package originates from site-packages."""
    import os
    import llm_training_stack
    package_file = Path(llm_training_stack.__file__).resolve()
    if os.environ.get("TRAIN_STACK_VERIFY_INSTALLED") == "1":
        assert "site-packages" in str(package_file), (
            f"Expected package to be loaded from site-packages, got: {package_file}"
        )
    assert package_file.exists()

