"""Tests for the train-stack Typer CLI commands."""

from typer.testing import CliRunner
from llm_training_stack.cli.main import app
from llm_training_stack.config.loader import ConfigLoader
from llm_training_stack.config.schema import TrainingJobConfig, TaskType, ModelConfig, DatasetConfig

runner = CliRunner()


def test_cli_inspect():
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 0
    assert "LLM Training Stack" in result.stdout
    assert "Platform" in result.stdout
    assert "System RAM" in result.stdout


def test_cli_compare(tmp_path):
    run_dir = tmp_path / "run_cli"
    run_dir.mkdir()

    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test_model"),
        dataset=DatasetConfig(dataset_name_or_path="test_dataset"),
    )
    from llm_training_stack.provenance.manifest import RunManifest
    m = RunManifest.create(cfg)
    m["final_metrics"] = {"final_loss": 1.25, "total_steps": 50, "total_duration_sec": 12.0}
    RunManifest.save(m, run_dir)

    result = runner.invoke(app, ["compare", str(run_dir)])
    assert result.exit_code == 0
    assert "Run ID" in result.stdout
    assert "1.25" in result.stdout


def test_cli_train(tmp_path):
    """Verifies that the Typer CLI 'train-stack train' command successfully launches training."""
    import json
    cfg_path = tmp_path / "train_cfg.json"
    out_dir = tmp_path / "cli_train_out"
    cfg_data = {
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM", "torch_dtype": "float32"},
        "dataset": {"dataset_name_or_path": "synthetic", "max_seq_length": 32, "train_sample_limit": 4},
        "hardware": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 1, "target_device": "cpu"},
        "logging": {"output_dir": str(out_dir), "save_steps": 2},
        "max_steps": 2,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    result = runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "Job Finished Successfully" in result.stdout
    assert (out_dir / "manifest.json").exists()


def test_cli_train_with_resume_argument(tmp_path):
    """Verifies that 'train-stack train --config ... --resume ...' correctly resumes training from checkpoint."""
    import json
    cfg_path = tmp_path / "train_initial_cfg.json"
    out_dir = tmp_path / "cli_train_resume_out"
    cfg_data = {
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM", "torch_dtype": "float32"},
        "dataset": {"dataset_name_or_path": "synthetic", "max_seq_length": 32, "train_sample_limit": 4},
        "hardware": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 1, "target_device": "cpu"},
        "logging": {"output_dir": str(out_dir), "save_steps": 2},
        "max_steps": 2,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    res1 = runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert res1.exit_code == 0
    assert (out_dir / "checkpoint-2").exists()

    # Resume run configuration targeting step 4
    cfg_res_data = dict(cfg_data)
    cfg_res_data["max_steps"] = 4
    cfg_res_data["dataset"]["train_sample_limit"] = 8
    cfg_res_path = tmp_path / "train_resume_cfg.json"
    with open(cfg_res_path, "w", encoding="utf-8") as f:
        json.dump(cfg_res_data, f)

    res2 = runner.invoke(app, [
        "train",
        "--config", str(cfg_res_path),
        "--resume", str(out_dir / "checkpoint-2"),
    ])
    assert res2.exit_code == 0
    assert "Job Finished Successfully" in res2.stdout
    assert "Total Steps: 4" in res2.stdout
    assert (out_dir / "checkpoint-4").exists()


def test_cli_resume(tmp_path):
    """Verifies that the CLI 'resume' command correctly inspects a saved checkpoint bundle."""
    from llm_training_stack.checkpoints.manager import CheckpointManager
    from transformers import LlamaConfig, LlamaForCausalLM
    import torch

    cp_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(cp_dir)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
    model = LlamaForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda x: 1.0)
    cp_path = mgr.save_checkpoint(step=5, epoch=1.0, model=model, optimizer=opt, lr_scheduler=sched, loss=0.45)

    result = runner.invoke(app, ["resume", "--checkpoint", str(cp_path)])
    assert result.exit_code == 0
    assert "Checkpoint Inspection" in result.stdout
    assert "Step: 5" in result.stdout
    assert "Is Complete Bundle: True" in result.stdout


def test_cli_invalid_config_rejection(tmp_path):
    """Verifies that the CLI rejects non-existent or invalid configuration files."""
    result = runner.invoke(app, ["train", "--config", str(tmp_path / "nonexistent.yaml")])
    assert result.exit_code != 0


def test_cli_eval(tmp_path):
    """Verifies that the Typer CLI 'train-stack eval' command executes and outputs metrics."""
    import json
    data_file = tmp_path / "eval_data.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "Hello world from causal language model evaluation."}) + "\n")
        f.write(json.dumps({"text": "Testing perplexity and loss calculation via CLI."}) + "\n")

    result = runner.invoke(app, [
        "eval",
        "--model", "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "--dataset", str(data_file),
        "--max-seq-length", "32",
    ])
    assert result.exit_code == 0
    assert "LLM Training Stack" in result.stdout
    assert "Cross-Entropy Loss" in result.stdout
    assert "Perplexity" in result.stdout


def test_cli_preflight(tmp_path):
    """Verifies that Typer CLI 'train-stack preflight' runs isolated probe and formats results table."""
    import json
    cfg_path = tmp_path / "preflight_cfg.json"
    cfg_data = {
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
        "dataset": {"dataset_name_or_path": "synthetic", "max_seq_length": 32},
        "hardware": {"per_device_train_batch_size": 2, "target_device": "cpu"},
        "logging": {"output_dir": str(tmp_path / "out")},
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    result = runner.invoke(app, ["preflight", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "Preflight Memory & Probe Verdict" in result.stdout
    assert "Probe Verdict" in result.stdout
    assert "Measured Peak Memory" in result.stdout


def test_cli_lifecycle_commands(tmp_path):
    """Verifies CLI status, logs, and cancel commands."""
    import json
    from llm_training_stack.lifecycle.manager import JobRecord, JobStatus

    run_dir = tmp_path / "cli_lifecycle_run"
    run_dir.mkdir()
    record = JobRecord(
        job_id="job_test_cli_123",
        status=JobStatus.RUNNING,
        task_type=TaskType.SFT,
        created_at="2026-10-06T00:00:00Z",
        run_dir=str(run_dir),
        config={"task_type": "sft"},
        total_steps=100,
        current_step=42,
        latest_loss=1.234,
    )
    record.save(run_dir)

    # Write events.jsonl
    events_file = run_dir / "events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 42, "loss": 1.234, "tokens_per_second": 500.0, "timestamp": "2026-10-06T00:01:00Z"}) + "\n")

    # 1. status
    res_status = runner.invoke(app, ["status", str(run_dir)])
    assert res_status.exit_code == 0
    assert "job_test_cli_123" in res_status.stdout
    assert "RUNNING" in res_status.stdout
    assert "42 / 100" in res_status.stdout

    # 2. logs
    res_logs = runner.invoke(app, ["logs", str(run_dir), "--tail", "10"])
    assert res_logs.exit_code == 0
    assert "1.234" in res_logs.stdout

    # 3. cancel
    res_cancel = runner.invoke(app, ["cancel", str(run_dir)])
    assert res_cancel.exit_code == 0
    assert "cancellation" in res_cancel.stdout.lower() or "cancelled" in res_cancel.stdout.lower()
    assert (run_dir / "cancel.token").exists()


def test_cli_submit_and_status_lookup(tmp_path):
    """CEO Delta 11: CLI submits in background and exits while job progresses, jobID lookup across process."""
    import json
    import re
    cfg_path = tmp_path / "submit_cfg.json"
    out_dir = tmp_path / "cli_submit_out"
    cfg_data = {
        "task_type": "sft",
        "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
        "dataset": {"dataset_name_or_path": "synthetic", "max_seq_length": 32, "train_sample_limit": 4},
        "hardware": {"per_device_train_batch_size": 2, "target_device": "cpu"},
        "logging": {"output_dir": str(out_dir)},
        "max_steps": 2,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    res_sub = runner.invoke(app, ["submit", "--config", str(cfg_path), "--background"])
    assert res_sub.exit_code == 0
    assert "Job Submitted" in res_sub.stdout
    assert "ID=" in res_sub.stdout

    match = re.search(r"ID=([^\s]+)", res_sub.stdout)
    assert match is not None
    job_id = match.group(1)

    res_stat = runner.invoke(app, ["status", job_id])
    assert res_stat.exit_code == 0
    assert job_id in res_stat.stdout


def test_cli_terminal_job_cancellation_wording(tmp_path):
    """CEO Delta 24: CLI cancel on already-terminal jobs reports truthful terminal status without falsely claiming cancellation."""
    from llm_training_stack.lifecycle.manager import JobRecord, JobStatus

    # 1. COMPLETED job fixture
    completed_dir = tmp_path / "job_completed_fixture"
    completed_dir.mkdir()
    rec_completed = JobRecord(
        job_id="job_completed_123",
        status=JobStatus.COMPLETED,
        task_type=TaskType.SFT,
        created_at="2026-10-06T00:00:00Z",
        finished_at="2026-10-06T00:05:00Z",
        run_dir=str(completed_dir),
        config={"task_type": "sft"},
        total_steps=10,
        current_step=10,
        latest_loss=0.5,
    )
    rec_completed.save(completed_dir)

    res_completed = runner.invoke(app, ["cancel", str(completed_dir)])
    assert res_completed.exit_code == 0
    assert "had already completed prior to cancellation" in res_completed.stdout
    assert "COMPLETED" in res_completed.stdout
    assert "cancelled. Status: COMPLETED" not in res_completed.stdout

    # 2. FAILED job fixture
    failed_dir = tmp_path / "job_failed_fixture"
    failed_dir.mkdir()
    rec_failed = JobRecord(
        job_id="job_failed_123",
        status=JobStatus.FAILED,
        task_type=TaskType.SFT,
        created_at="2026-10-06T00:00:00Z",
        finished_at="2026-10-06T00:02:00Z",
        run_dir=str(failed_dir),
        config={"task_type": "sft"},
        total_steps=10,
        current_step=3,
        error="CUDA OOM or synthetic failure",
    )
    rec_failed.save(failed_dir)

    res_failed = runner.invoke(app, ["cancel", str(failed_dir)])
    assert res_failed.exit_code == 0
    assert "had already failed prior to cancellation" in res_failed.stdout
    assert "FAILED" in res_failed.stdout
    assert "cancellation requested. Status: FAILED" not in res_failed.stdout
    assert "cancelled. Status: FAILED" not in res_failed.stdout






