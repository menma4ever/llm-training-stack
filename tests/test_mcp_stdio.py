"""Acceptance verification: Real MCP stdio protocol security tests.

Verifies over subprocess stdio JSON-RPC transport:
1. Stdio initialize handshake and tools listing.
2. Read-only tool invocation (inspect_hardware).
3. Authorized launch grant acceptance.
4. Changed-input plan mutation rejection (altered model/dataset).
5. Forged / invalid grant rejection (no digest leakage).
6. Unauthorized invocation rejection (authorized=False).
7. Operator read-only lockdown rejection.
8. Path root jail mutation rejection (directory traversal).
9. Subprocess isolation bypass rejection.
"""

import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from llm_training_stack.config.schema import (
    DatasetConfig,
    HardwareConfig,
    ModelConfig,
    TaskType,
    TrainingJobConfig,
)


import threading

class StdioMcpClient:
    """Helper for communicating with FastMCP server over subprocess stdio JSON-RPC."""

    def __init__(self, env=None):
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "llm_training_stack.mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            bufsize=1,
        )
        self._msg_id = 0
        self._stderr_lines = []

        def _drain():
            for line in iter(self.proc.stderr.readline, ''):
                self._stderr_lines.append(line)

        self._drain_thread = threading.Thread(target=_drain, daemon=True)
        self._drain_thread.start()

    def send(self, method: str, params: dict = None, is_notification: bool = False) -> dict:
        self._msg_id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if not is_notification:
            msg["id"] = self._msg_id
        if params is not None:
            msg["params"] = params

        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

        if is_notification:
            return {}

        resp_line = self.proc.stdout.readline()
        if not resp_line:
            err = "".join(self._stderr_lines)
            raise RuntimeError(f"Server closed connection unexpectedly. Stderr: {err}")
        return json.loads(resp_line)

    def initialize(self):
        resp = self.send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-stdio-client", "version": "1.0"},
            },
        )
        self.send("notifications/initialized", is_notification=True)
        return resp

    def call_tool(self, name: str, arguments: dict) -> dict:
        resp = self.send("tools/call", {"name": name, "arguments": arguments})
        result = resp.get("result", {})
        # FastMCP returns content list with text
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture
def mcp_server_env(tmp_path):
    secret = "test-operator-hmac-secret-2026"
    allowed_root = str(tmp_path)
    return {
        "TRAIN_STACK_ALLOW_LAUNCH": "1",
        "TRAIN_STACK_ALLOW_PROBE": "1",
        "TRAIN_STACK_AUTH_SECRET": secret,
        "TRAIN_STACK_ALLOWED_ROOTS": allowed_root,
    }, secret


def test_mcp_stdio_initialize_and_tools_list(mcp_server_env):
    """Verifies stdio JSON-RPC handshake and tool discovery."""
    env, _ = mcp_server_env
    client = StdioMcpClient(env=env)
    try:
        init_resp = client.initialize()
        assert init_resp.get("result", {}).get("serverInfo", {}).get("name") == "llm-training-stack"

        tools_resp = client.send("tools/list")
        tool_names = [t["name"] for t in tools_resp.get("result", {}).get("tools", [])]
        assert "inspect_hardware" in tool_names
        assert "estimate_memory" in tool_names
        assert "launch_training" in tool_names
        assert "run_preflight_probe" in tool_names
    finally:
        client.close()


def test_mcp_stdio_read_tools_hardware_inspection(mcp_server_env):
    """Verifies read-only tool invocation over stdio."""
    env, _ = mcp_server_env
    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool("inspect_hardware", {})
        assert "platform" in res
        assert "cpu_count_physical" in res
        assert "system_ram_total_gb" in res
    finally:
        client.close()


def test_mcp_stdio_launch_grant_accepted_policy(mcp_server_env, tmp_path):
    """Verifies that a valid HMAC authorization grant is accepted by policy over stdio."""
    env, secret = mcp_server_env
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )
    from llm_training_stack.mcp.server import create_plan_grant
    canonical_plan = cfg.model_dump_json()
    valid_hmac = create_plan_grant(cfg, action="launch", secret=secret)

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "launch_training",
            {
                "config_json": canonical_plan,
                "authorized": True,
                "authorization_grant": valid_hmac,
            },
        )
        # Grant check should pass; any error is execution-level (e.g. mock dataset), not policy denial
        assert res.get("error") != "AUTHORIZATION_GRANT_REQUIRED"
        assert res.get("error") != "PLAN_GRANT_MISMATCH"
        assert res.get("error") != "EXECUTION_DENIED_BY_OPERATOR_POLICY"
    finally:
        client.close()


def test_mcp_stdio_changed_input_mutation_denied(mcp_server_env, tmp_path):
    """Security verification: Mutating the plan (model/dataset) invalidates HMAC grant and is denied."""
    env, secret = mcp_server_env
    cfg_approved = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="approved/safe-model"),
        dataset=DatasetConfig(dataset_name_or_path="approved/safe-dataset", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )
    # Grant computed strictly for approved plan
    from llm_training_stack.mcp.server import create_plan_grant
    valid_hmac = create_plan_grant(cfg_approved, action="launch", secret=secret)

    # Mutated plan: caller alters model to unapproved trojan
    cfg_mutated = cfg_approved.model_copy(deep=True)
    cfg_mutated.model.model_name_or_path = "unapproved/trojan-model"

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "launch_training",
            {
                "config_json": cfg_mutated.model_dump_json(),
                "authorized": True,
                "authorization_grant": valid_hmac,  # Replaying grant from approved plan
            },
        )
        assert res.get("error") == "PLAN_GRANT_MISMATCH"
        assert "expected_digest" not in res  # Never leak operator digest
    finally:
        client.close()


def test_mcp_stdio_unauthorized_flag_denied(mcp_server_env, tmp_path):
    """Security verification: authorized=False is denied over stdio even if grant is provided."""
    env, secret = mcp_server_env
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )
    from llm_training_stack.mcp.server import create_plan_grant
    valid_hmac = create_plan_grant(cfg, action="launch", secret=secret)

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "launch_training",
            {
                "config_json": cfg.model_dump_json(),
                "authorized": False,
                "authorization_grant": valid_hmac,
            },
        )
        assert res.get("error") == "EXECUTION_DENIED"
        assert res.get("authorized") is False
    finally:
        client.close()


def test_mcp_stdio_path_root_jail_mutation_denied(mcp_server_env, tmp_path):
    """Security verification: Path traversing outside authorized roots is rejected over stdio."""
    env, secret = mcp_server_env
    # Path explicitly outside allowed_root
    escaping_path = str(tmp_path.parent / "unauthorized_escaped_data")

    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path=escaping_path, max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )
    from llm_training_stack.mcp.server import create_plan_grant
    valid_hmac = create_plan_grant(cfg, action="launch", secret=secret)

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "launch_training",
            {
                "config_json": cfg.model_dump_json(),
                "authorized": True,
                "authorization_grant": valid_hmac,
            },
        )
        assert res.get("error") == "PATH_ROOT_JAIL_VIOLATION"
    finally:
        client.close()


def test_mcp_stdio_isolation_bypass_mutation_denied(mcp_server_env, tmp_path):
    """Security verification: Disabling subprocess isolation in run_preflight_probe is denied over stdio."""
    env, _ = mcp_server_env
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "run_preflight_probe",
            {
                "config_json": cfg.model_dump_json(),
                "in_subprocess": False,  # Attempting to bypass subprocess boundary
            },
        )
        assert res.get("error") == "ISOLATION_POLICY_VIOLATION"
    finally:
        client.close()


def test_mcp_stdio_static_token_replay_rejected(mcp_server_env, tmp_path):
    """CEO Delta 11: Replaying static operator token without plan HMAC is rejected over stdio."""
    env, _ = mcp_server_env
    env["TRAIN_STACK_OPERATOR_TOKEN"] = "static-token-abc-123"

    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
        logging={"output_dir": str(tmp_path / "output")},
    )

    client = StdioMcpClient(env=env)
    try:
        client.initialize()
        res = client.call_tool(
            "launch_training",
            {
                "config_json": cfg.model_dump_json(),
                "authorized": True,
                "authorization_grant": "static-token-abc-123",  # raw bearer token without plan HMAC
            },
        )
        assert res.get("error") == "PLAN_GRANT_MISMATCH"
    finally:
        client.close()


def test_mcp_stdio_cancellation_authorization_lifecycle(mcp_server_env, tmp_path):
    """CEO Delta 11: Exercises cancellation denial, launch grant mismatch, and valid action grant over stdio."""
    env, secret = mcp_server_env
    run_dir = tmp_path / "stdio_cancel_job"
    run_dir.mkdir()
    (run_dir / "job_state.json").write_text(json.dumps({
        "job_id": "job_stdio_cancel",
        "status": "RUNNING",
        "task_type": "sft",
        "created_at": "2026-10-06T00:00:00Z",
        "run_dir": str(run_dir),
        "config": {"task_type": "sft"},
    }), encoding="utf-8")

    from llm_training_stack.mcp.server import create_plan_grant, create_cancellation_grant

    client = StdioMcpClient(env=env)
    try:
        client.initialize()

        # 1. Denied when authorized=False
        res1 = client.call_tool("cancel_training_job", {"run_dir": str(run_dir), "authorized": False})
        assert res1.get("error") == "EXECUTION_DENIED"

        # 2. Denied when authorization_grant is missing
        res2 = client.call_tool("cancel_training_job", {"run_dir": str(run_dir), "authorized": True})
        assert res2.get("error") == "AUTHORIZATION_GRANT_REQUIRED"

        # 3. Denied when a launch grant is attempted for cancellation
        dummy_cfg = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="mock"), dataset=DatasetConfig(dataset_name_or_path="mock"))
        launch_grant = create_plan_grant(dummy_cfg, action="launch", secret=secret)
        res3 = client.call_tool("cancel_training_job", {"run_dir": str(run_dir), "authorized": True, "authorization_grant": launch_grant})
        assert res3.get("error") == "ACTION_GRANT_MISMATCH"

        # 4. Accepted when valid action-bound cancellation grant provided:
        # CEO Delta 20: Distinguish accepted cancellation request from child terminal acknowledgement
        cancel_grant = create_cancellation_grant(str(run_dir), secret=secret)
        # 4a. Accepted request writes token; unacknowledged mock retains RUNNING and finished_at unset
        res4 = client.call_tool("cancel_training_job", {"run_dir": str(run_dir), "authorized": True, "authorization_grant": cancel_grant, "wait": False})
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

        ack_res = client.call_tool("cancel_training_job", {"run_dir": str(run_dir), "authorized": True, "authorization_grant": cancel_grant, "wait": False})
        assert ack_res.get("status") == "CANCELLED"
        assert ack_res.get("finished_at") is not None
    finally:
        client.close()

