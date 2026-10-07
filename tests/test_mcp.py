"""Tests for safe Model Context Protocol (MCP) server adapter and security boundaries."""

import hashlib
import hmac
import json
from pathlib import Path
from llm_training_stack.config.loader import ConfigLoader
from llm_training_stack.mcp.server import inspect_hardware, estimate_memory, launch_training, run_preflight_probe, audit_tokenizer


def test_mcp_inspect_hardware_tool():
    res_str = inspect_hardware()
    data = json.loads(res_str)
    assert "platform" in data
    assert "system_ram_total_gb" in data


def test_mcp_estimate_memory_tool():
    res_str = estimate_memory(total_params=500_000_000, task_type="lora")
    data = json.loads(res_str)
    assert data["total_params"] == 500_000_000
    assert "estimated_total_gb" in data
    assert data["trainable_percent"] < 1.0


def test_mcp_launch_unauthorized_rejection():
    """Security verification: launch_training MUST reject execution when authorized=False."""
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(config_json=json.dumps(cfg), authorized=False)
    data = json.loads(res_str)
    assert data.get("error") == "EXECUTION_DENIED"
    assert data.get("authorized") is False


def test_mcp_launch_operator_policy_rejection(monkeypatch):
    """Security verification: authorized=True from agent is rejected if server operator policy forbids launch."""
    monkeypatch.delenv("TRAIN_STACK_ALLOW_LAUNCH", raising=False)
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(config_json=json.dumps(cfg), authorized=True)
    data = json.loads(res_str)
    assert data.get("error") == "EXECUTION_DENIED_BY_OPERATOR_POLICY"
    assert data.get("operator_policy") == "READ_ONLY"


def test_mcp_trust_remote_code_rejection(monkeypatch):
    """Security verification: trust_remote_code=True is rejected unless explicitly enabled by server operator."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.delenv("TRAIN_STACK_ALLOW_REMOTE_CODE", raising=False)
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock", "trust_remote_code": True},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(config_json=json.dumps(cfg), authorized=True)
    data = json.loads(res_str)
    assert data.get("error") == "SECURITY_POLICY_VIOLATION"


def test_mcp_run_preflight_probe_operator_rejection(monkeypatch):
    """Security verification: run_preflight_probe is rejected if TRAIN_STACK_ALLOW_PROBE!=1."""
    monkeypatch.delenv("TRAIN_STACK_ALLOW_PROBE", raising=False)
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = run_preflight_probe(config_json=json.dumps(cfg))
    data = json.loads(res_str)
    assert data.get("error") == "EXECUTION_DENIED_BY_OPERATOR_POLICY"
    assert data.get("operator_policy") == "READ_ONLY"


def test_mcp_probe_remote_code_denial_before_model_load(monkeypatch):
    """Strict verification: Prove from_pretrained is NEVER called when remote code is rejected."""
    from transformers import AutoModelForCausalLM

    def fake_from_pretrained(*args, **kwargs):
        raise AssertionError("AutoModelForCausalLM.from_pretrained MUST NOT be called when security policy denies request!")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setenv("TRAIN_STACK_ALLOW_PROBE", "1")
    monkeypatch.delenv("TRAIN_STACK_ALLOW_REMOTE_CODE", raising=False)

    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "malicious/untrusted-model", "trust_remote_code": True},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = run_preflight_probe(config_json=json.dumps(cfg))
    data = json.loads(res_str)
    assert data.get("error") == "SECURITY_POLICY_VIOLATION"


def test_mcp_launch_grant_required_by_default(monkeypatch):
    """B-1 Security: Launch without grant is rejected by default (AUTHORIZATION_GRANT_REQUIRED)."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.delenv("TRAIN_STACK_DEV_BYPASS_GRANT", raising=False)

    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(
        config_json=json.dumps(cfg),
        authorized=True,
        authorization_grant=None,
    )
    data = json.loads(res_str)
    assert data.get("error") == "AUTHORIZATION_GRANT_REQUIRED"


def test_mcp_launch_plan_grant_mismatch(monkeypatch):
    """B-1 Security: Invalid grant is rejected and NEVER echoes expected digest (prevents replay)."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_AUTH_SECRET", "super-secret-operator-key-999")

    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(
        config_json=json.dumps(cfg),
        authorized=True,
        authorization_grant="forged-client-hash-12345",
    )
    data = json.loads(res_str)
    assert data.get("error") == "PLAN_GRANT_MISMATCH"
    assert "expected_digest" not in data  # Proves no digest leak occurs


def test_mcp_launch_valid_hmac_grant_validation(monkeypatch):
    """B-1 Security: Valid HMAC grant computed with secret passes grant check."""
    secret = "production-operator-key-42"
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_AUTH_SECRET", secret)

    cfg_dict = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    typed_cfg = ConfigLoader.load_from_dict(cfg_dict)
    from llm_training_stack.mcp.server import validate_mcp_policy, create_plan_grant
    valid_hmac = create_plan_grant(typed_cfg, action="launch", secret=secret)

    # Pass valid HMAC grant
    policy_err = validate_mcp_policy(
        typed_cfg,
        action="launch",
        authorized=True,
        authorization_grant=valid_hmac,
    )
    assert policy_err is None


def test_mcp_launch_operator_token_validation(monkeypatch):
    """CEO Delta 10/11: Raw static operator token without plan HMAC is rejected."""
    token = "operator-token-xyz-777"
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_OPERATOR_TOKEN", token)

    cfg_dict = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    typed_cfg = ConfigLoader.load_from_dict(cfg_dict)

    from llm_training_stack.mcp.server import validate_mcp_policy, create_plan_grant
    # 1. Raw static token rejected
    policy_err = validate_mcp_policy(
        typed_cfg,
        action="launch",
        authorized=True,
        authorization_grant=token,
    )
    assert policy_err is not None
    assert policy_err["error"] == "PLAN_GRANT_MISMATCH"

    # 2. Plan-bound HMAC with operator token passes
    valid_grant = create_plan_grant(typed_cfg, action="launch", secret=token)
    ok_err = validate_mcp_policy(
        typed_cfg,
        action="launch",
        authorized=True,
        authorization_grant=valid_grant,
    )
    assert ok_err is None


def test_mcp_cancellation_authorization(tmp_path, monkeypatch):
    """CEO Delta 11: cancel_training_job validates mutating authorization grants."""
    secret = "cancel-secret-key-999"
    monkeypatch.setenv("TRAIN_STACK_AUTH_SECRET", secret)
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_ALLOWED_ROOTS", str(tmp_path))

    run_dir = tmp_path / "cancel_run"
    run_dir.mkdir()
    (run_dir / "job_state.json").write_text(json.dumps({
        "job_id": "job_cancel_test",
        "status": "RUNNING",
        "task_type": "sft",
        "created_at": "2026-10-06T00:00:00Z",
        "run_dir": str(run_dir),
        "config": {"task_type": "sft"},
    }), encoding="utf-8")

    from llm_training_stack.mcp.server import cancel_training_job, create_cancellation_grant, create_plan_grant
    from llm_training_stack.config.schema import TrainingJobConfig, TaskType, ModelConfig, DatasetConfig

    # 1. Denial when authorized=False
    res1 = json.loads(cancel_training_job(run_dir=str(run_dir), authorized=False))
    assert res1["error"] == "EXECUTION_DENIED"

    # 2. Denial when grant is missing
    res2 = json.loads(cancel_training_job(run_dir=str(run_dir), authorized=True, authorization_grant=None))
    assert res2["error"] == "AUTHORIZATION_GRANT_REQUIRED"

    # 3. Denial when launch grant used for cancellation
    dummy_cfg = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m"), dataset=DatasetConfig(dataset_name_or_path="d"))
    launch_grant = create_plan_grant(dummy_cfg, action="launch", secret=secret)
    res3 = json.loads(cancel_training_job(run_dir=str(run_dir), authorized=True, authorization_grant=launch_grant))
    assert res3["error"] == "ACTION_GRANT_MISMATCH"

    # 4. Success when valid action-bound cancellation grant provided:
    # CEO Delta 20: Distinguish accepted cancellation request from child terminal acknowledgement
    cancel_grant = create_cancellation_grant(str(run_dir), secret=secret)
    # 4a. Accepted cancellation request writes token; without child worker acknowledgement, retains RUNNING
    res4 = json.loads(cancel_training_job(run_dir=str(run_dir), authorized=True, authorization_grant=cancel_grant, wait=False))
    assert (run_dir / "cancel.token").exists()
    assert res4.get("cancellation_token_created") is True
    assert res4.get("status") == "RUNNING"
    assert res4.get("finished_at") is None

    # 4b. When child worker acknowledges cancellation, terminal status is CANCELLED
    rec = json.loads((run_dir / "job_state.json").read_text(encoding="utf-8"))
    rec["status"] = "CANCELLED"
    rec["finished_at"] = "2026-10-06T00:00:01Z"
    rec["error"] = "Cancelled by operator request."
    (run_dir / "job_state.json").write_text(json.dumps(rec), encoding="utf-8")

    ack_res = json.loads(cancel_training_job(run_dir=str(run_dir), authorized=True, authorization_grant=cancel_grant, wait=False))
    assert ack_res.get("status") == "CANCELLED"
    assert ack_res.get("finished_at") is not None


def test_mcp_cancellation_immutable_target_and_hostile_verifications(tmp_path, monkeypatch):
    """Hostile technical verification: Enforces immutable target resolution, cross-job grant replay prevention, and path jail bounds."""
    secret = "operator-secret-hostile-555"
    monkeypatch.setenv("TRAIN_STACK_AUTH_SECRET", secret)
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_ALLOWED_ROOTS", str(tmp_path))

    from llm_training_stack.mcp.server import cancel_training_job, create_cancellation_grant
    from llm_training_stack.lifecycle.manager import _register_job

    # Create Job A
    job_a_dir = tmp_path / "job_a"
    job_a_dir.mkdir()
    (job_a_dir / "job_state.json").write_text(json.dumps({
        "job_id": "job_id_alpha",
        "status": "RUNNING",
        "task_type": "sft",
        "created_at": "2026-10-06T00:00:00Z",
        "run_dir": str(job_a_dir),
        "config": {"task_type": "sft"},
    }), encoding="utf-8")
    _register_job("job_id_alpha", job_a_dir)

    # Create Job B
    job_b_dir = tmp_path / "job_b"
    job_b_dir.mkdir()
    (job_b_dir / "job_state.json").write_text(json.dumps({
        "job_id": "job_id_beta",
        "status": "RUNNING",
        "task_type": "sft",
        "created_at": "2026-10-06T00:00:00Z",
        "run_dir": str(job_b_dir),
        "config": {"task_type": "sft"},
    }), encoding="utf-8")
    _register_job("job_id_beta", job_b_dir)

    # 1. Hostile Cross-Job Grant Replay: Use grant signed for Job A to cancel Job B
    grant_a = create_cancellation_grant(str(job_a_dir), secret=secret)
    replay_res = json.loads(cancel_training_job(run_dir=str(job_b_dir), authorized=True, authorization_grant=grant_a))
    assert replay_res.get("error") == "ACTION_GRANT_MISMATCH"
    assert not (job_b_dir / "cancel.token").exists()

    # 2. Hostile Path Jail Violation: Attempt to cancel a directory outside allowed roots
    jail_res = json.loads(cancel_training_job(run_dir="C:/Windows/System32", authorized=True, authorization_grant=grant_a))
    assert jail_res.get("error") in ("PATH_ROOT_JAIL_VIOLATION", "JOB_NOT_FOUND")

    # 3. Nonexistent job target
    missing_res = json.loads(cancel_training_job(run_dir=str(tmp_path / "nonexistent_job_xyz"), authorized=True, authorization_grant=grant_a))
    assert missing_res.get("error") == "JOB_NOT_FOUND"

    # 4. Immutable Target Mutation: Cancel Job A by its Job ID alias
    # Server resolves job_id_alpha to job_a_dir, verifies grant, and mutates strictly on resolved canonical dir
    grant_alias_a = create_cancellation_grant("job_id_alpha", secret=secret)
    cancel_alias_res = json.loads(cancel_training_job(run_dir="job_id_alpha", authorized=True, authorization_grant=grant_alias_a, wait=False))
    assert (job_a_dir / "cancel.token").exists()
    assert cancel_alias_res.get("cancellation_token_created") is True
    # Without active child worker to acknowledge, status on disk correctly retains RUNNING
    assert cancel_alias_res.get("status") == "RUNNING"
    assert cancel_alias_res.get("finished_at") is None
    assert json.loads((job_a_dir / "job_state.json").read_text(encoding="utf-8"))["status"] == "RUNNING"

    # 5. Hostile Dynamic Alias Retarget Negative:
    # Register mutable alias 'dynamic_job_alias' initially pointing to Job A
    _register_job("dynamic_job_alias", job_a_dir)
    grant_before_retarget = create_cancellation_grant(str(job_a_dir), secret=secret)
    grant_raw_alias = create_cancellation_grant("dynamic_job_alias", secret=secret)

    # Retarget the alias to point to Job B
    _register_job("dynamic_job_alias", job_b_dir)

    # Attempt 5a: Using grant signed for Job A before retarget must be rejected on Job B
    retarget_res1 = json.loads(cancel_training_job(run_dir="dynamic_job_alias", authorized=True, authorization_grant=grant_before_retarget))
    assert retarget_res1.get("error") == "ACTION_GRANT_MISMATCH"
    assert not (job_b_dir / "cancel.token").exists()
    assert json.loads((job_b_dir / "job_state.json").read_text(encoding="utf-8"))["status"] == "RUNNING"

    # Attempt 5b: Using grant signed for raw mutable alias string must be rejected
    retarget_res2 = json.loads(cancel_training_job(run_dir="dynamic_job_alias", authorized=True, authorization_grant=grant_raw_alias))
    assert retarget_res2.get("error") == "ACTION_GRANT_MISMATCH"
    assert not (job_b_dir / "cancel.token").exists()
    assert json.loads((job_b_dir / "job_state.json").read_text(encoding="utf-8"))["status"] == "RUNNING"


def test_mcp_path_root_jail_rejection(monkeypatch):
    """m-1 Security: Paths escaping authorized filesystem roots are rejected with PATH_ROOT_JAIL_VIOLATION."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_LAUNCH", "1")
    monkeypatch.setenv("TRAIN_STACK_DEV_BYPASS_GRANT", "1")
    monkeypatch.setenv("TRAIN_STACK_ALLOWED_ROOTS", "/var/isolated/training_jail")

    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "../../../../etc/unauthorized_model"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = launch_training(config_json=json.dumps(cfg), authorized=True)
    data = json.loads(res_str)
    assert data.get("error") == "PATH_ROOT_JAIL_VIOLATION"
    assert "outside authorized filesystem roots" in data.get("message", "")


def test_mcp_audit_tokenizer_operator_rejection(monkeypatch):
    """Security verification: audit_tokenizer rejects execution if TRAIN_STACK_ALLOW_PROBE!=1."""
    monkeypatch.delenv("TRAIN_STACK_ALLOW_PROBE", raising=False)
    res_str = audit_tokenizer(model_path="test/model")
    data = json.loads(res_str)
    assert data.get("error") == "EXECUTION_DENIED_BY_OPERATOR_POLICY"
    assert data.get("operator_policy") == "READ_ONLY"


def test_mcp_audit_tokenizer_path_not_found(monkeypatch):
    """Security verification: audit_tokenizer validates local filesystem paths before loading."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_PROBE", "1")
    res_str = audit_tokenizer(model_path="nonexistent/local/path/to/model")
    data = json.loads(res_str)
    assert data.get("error") == "PATH_NOT_FOUND"


def test_mcp_run_preflight_probe_isolation_bypass_rejection(monkeypatch):
    """Security verification: Caller cannot disable subprocess isolation over MCP."""
    monkeypatch.setenv("TRAIN_STACK_ALLOW_PROBE", "1")
    cfg = {
        "schema_version": "1.0.0",
        "task_type": "sft",
        "model": {"model_name_or_path": "mock"},
        "dataset": {"dataset_name_or_path": "mock"},
    }
    res_str = run_preflight_probe(config_json=json.dumps(cfg), in_subprocess=False)
    data = json.loads(res_str)
    assert data.get("error") == "ISOLATION_POLICY_VIOLATION"
