"""Tests for shared core durable job lifecycle management."""

import json
import pytest
import time
from pathlib import Path
from datetime import datetime, timezone
from llm_training_stack.config.schema import TrainingJobConfig, TaskType, ModelConfig, DatasetConfig, HardwareConfig, LoggingConfig
from llm_training_stack.lifecycle.manager import LifecycleManager, JobStatus, JobRecord


def test_lifecycle_submit_and_complete_job(tmp_path):
    out_dir = tmp_path / "lifecycle_run_complete"
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype="float32"),
        dataset=DatasetConfig(dataset_name_or_path="synthetic", max_seq_length=32, train_sample_limit=4),
        hardware=HardwareConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, target_device="cpu"),
        logging=LoggingConfig(output_dir=str(out_dir), save_steps=2),
        max_steps=2,
    )

    record = LifecycleManager.submit_job(cfg, run_in_background=False)
    assert record.status == JobStatus.COMPLETED
    assert record.current_step == 2
    assert record.latest_loss is not None
    assert record.finished_at is not None

    # Verify persistent state on disk
    loaded = LifecycleManager.get_job_status(out_dir)
    assert loaded.job_id == record.job_id
    assert loaded.status == JobStatus.COMPLETED

    # Verify structured logs inspection
    logs = LifecycleManager.get_job_logs(out_dir, tail_lines=10)
    assert len(logs) > 0


def test_lifecycle_failure_durable_recording(tmp_path):
    out_dir = tmp_path / "lifecycle_run_fail"
    # Provide nonexistent model path to trigger execution failure
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="nonexistent-model-xyz-12345", torch_dtype="float32"),
        dataset=DatasetConfig(dataset_name_or_path="synthetic", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, target_device="cpu"),
        logging=LoggingConfig(output_dir=str(out_dir)),
        max_steps=2,
    )

    record = LifecycleManager.submit_job(cfg, run_in_background=False)
    assert record.status == JobStatus.FAILED
    assert record.error is not None
    assert record.finished_at is not None

    # Verify persistent failure record on disk
    loaded = LifecycleManager.get_job_status(out_dir)
    assert loaded.status == JobStatus.FAILED
    assert loaded.error is not None


def test_lifecycle_cancellation(tmp_path):
    """Verifies active background child subprocess cancellation after observed progress,
    clean child exit, bounded cessation well before horizon, and preserved PID/logs."""
    import psutil
    out_dir = tmp_path / "lifecycle_run_cancel"
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype="float32"),
        dataset=DatasetConfig(dataset_name_or_path="synthetic", max_seq_length=32, train_sample_limit=8),
        hardware=HardwareConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, target_device="cpu"),
        logging=LoggingConfig(output_dir=str(out_dir), logging_steps=1, save_steps=2),
        max_steps=20,
    )

    record = LifecycleManager.submit_job(cfg, run_in_background=True)
    assert record.status == JobStatus.SUBMITTED

    # 1. Identify active child worker subprocess
    child_proc = None
    for _ in range(50):
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "run-worker" in cmd and str(out_dir) in cmd:
                    child_proc = p
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if child_proc:
            break
        time.sleep(0.1)

    assert child_proc is not None, "Child worker process was not spawned"
    child_pid = child_proc.pid

    # 2. Wait for observed active training progress (logs emitted)
    events_file = out_dir / "events.jsonl"
    progress_observed = False
    for _ in range(250):
        if events_file.exists():
            logs = LifecycleManager.get_job_logs(out_dir)
            if len(logs) >= 1:
                progress_observed = True
                break
        time.sleep(0.3)

    assert progress_observed, "Failed to observe active progress before requesting cancellation"

    # 3. Request active cancellation
    cancelled = LifecycleManager.cancel_job(out_dir)
    assert cancelled.status == JobStatus.CANCELLED
    assert (out_dir / "cancel.token").exists()

    # 4. Wait for child process to exit cleanly
    try:
        child_proc.wait(timeout=25)
    except psutil.TimeoutExpired:
        child_proc.kill()
        pytest.fail(f"Child worker {child_pid} did not exit within timeout following cancellation")

    assert not child_proc.is_running(), f"Child worker process {child_pid} still running"

    # 5. Verify bounded cessation well before horizon, state, metrics, and logs agreement
    final_record = LifecycleManager.get_job_status(out_dir)
    assert final_record.status == JobStatus.CANCELLED
    assert final_record.error is not None and "cancel" in final_record.error.lower()
    assert final_record.finished_at is not None

    logs = LifecycleManager.get_job_logs(out_dir)
    assert len(logs) > 0, "Preserved training event logs must not be empty"
    last_logged_step = logs[-1].get("step", 0)
    assert last_logged_step < cfg.max_steps, f"Job did not halt before horizon: step {last_logged_step} >= {cfg.max_steps}"

    # 6. Assert durable current_step, latest_loss, checkpoint, and manifest consistency (CEO Delta 19)
    assert final_record.current_step is not None and final_record.current_step > 0
    assert final_record.current_step == last_logged_step
    assert final_record.latest_loss is not None
    assert (out_dir / "manifest.json").exists()
    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data.get("task_type") == final_record.task_type.value


def test_lifecycle_cancellation_unacknowledged_retains_running(tmp_path):
    """CEO DELTA 20: Calling cancel_job on a RUNNING job when child does not acknowledge retains RUNNING and finished_at unset."""
    out_dir = tmp_path / "mock_running_job"
    out_dir.mkdir()
    record = JobRecord(
        job_id="job_unacknowledged_test",
        run_dir=str(out_dir),
        task_type=TaskType.SFT,
        config={"task_type": "sft"},
        status=JobStatus.RUNNING,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    record.save(out_dir)

    # Calling cancel_job with timeout and no child acknowledging must retain RUNNING with finished_at unset
    result = LifecycleManager.cancel_job(out_dir, wait=True, timeout=0.2)
    assert (out_dir / "cancel.token").exists()
    assert result.status == JobStatus.RUNNING
    assert result.finished_at is None

    # Disk status must remain RUNNING with finished_at unset
    disk_record = LifecycleManager.get_job_status(out_dir)
    assert disk_record.status == JobStatus.RUNNING
    assert disk_record.finished_at is None

