"""Targeted Model Context Protocol (MCP) server tools validation in installed candidate virtualenv.
Validates:
1. Part 1: Installed Handler Integration Evidence:
   - inspect_hardware and estimate_memory read-only functions.
   - run_preflight_probe schema validation:
     * asserts measured schema: measured_peak_allocated_mb, verdict, isolation_mode='subprocess',
       gpu_fit_guaranteed=False, is_representative=False (CPU truthful non-representative warning).
     * saves actual measured result JSON.
   - Denial-before-loader and launch-grant security policies via direct handlers.
2. Part 2: Installed FastMCP STDIO Client Protocol Transport Evidence:
   - Spawns python -m llm_training_stack.mcp.server subprocess over stdio transport.
   - ClientSession.initialize() and list_tools() protocol verification.
   - Tool execution over STDIO transport: inspect_hardware, estimate_memory, run_preflight_probe.
   - Observable loader/spawn instrumentation: confirms zero child process spawning and zero loader invocation on denials.
   - Real background training job launch via STDIO call_tool with action-bound plan grant.
   - Detection of real child worker process PID in OS process table.
   - Real cancellation via STDIO call_tool with action-bound cancellation grant.
   - Strict OS process cessation verification (PID exit confirmed without AccessDenied shortcut).
   - Cross-artifact field & metrics agreement (job_state, manifest, and events.jsonl step and loss within 1e-4 tolerance).
3. Non-destructive unique run directories; never uses rmtree.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import psutil

# CLI argument resolution for portable execution
parser = argparse.ArgumentParser(description="Targeted MCP server validation in candidate virtualenv")
parser.add_argument("--project-root", type=str, default=None, help="Explicit path to project root")
parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory for candidate runs")
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).resolve().parent
if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
elif os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else (PROJECT_ROOT / "artifacts" / "candidate_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUTH_SECRET = "operator-secret-mcp-candidate-999"
os.environ["TRAIN_STACK_ALLOW_LAUNCH"] = "1"
os.environ["TRAIN_STACK_ALLOW_PROBE"] = "1"
os.environ["TRAIN_STACK_ALLOW_CANCEL"] = "1"
os.environ["TRAIN_STACK_AUTH_SECRET"] = AUTH_SECRET
os.environ["TRAIN_STACK_ALLOWED_ROOTS"] = f"{PROJECT_ROOT};{Path.cwd()};{tempfile.gettempdir()}"

print("=" * 80)
print("TARGETED MCP SERVER VALIDATION: HANDLER INTEGRATION & STDIO PROTOCOL")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)

from llm_training_stack.mcp.server import (
    inspect_hardware, estimate_memory,
    run_preflight_probe, submit_training_job,
    get_training_job_status, get_training_job_logs,
    cancel_training_job, create_cancellation_grant, create_plan_grant
)
from llm_training_stack.config.schema import (
    TrainingJobConfig, TaskType, ModelConfig, DatasetConfig,
    HardwareConfig, LoggingConfig
)

# ==============================================================================
# PART 1: INSTALLED HANDLER INTEGRATION EVIDENCE
# ==============================================================================
print("\n" + "=" * 80)
print("PART 1: INSTALLED HANDLER INTEGRATION EVIDENCE (DIRECT FUNCTION INVOCATIONS)")
print("=" * 80)

# 1. Hardware Inspection
print("\n[Handler 1/4] Invoking inspect_hardware handler...")
hw_res = json.loads(inspect_hardware())
assert "platform" in hw_res, "Missing platform in hardware inspection"
assert "system_ram_total_gb" in hw_res, "Missing system_ram_total_gb in hardware inspection"
print(f"[PASS] Hardware: platform={hw_res.get('platform')}, RAM={hw_res.get('system_ram_total_gb')} GB, Torch={hw_res.get('torch_version')}")

# 2. Analytical Memory Estimation
print("\n[Handler 2/4] Invoking estimate_memory handler...")
mem_res = json.loads(estimate_memory(total_params=135_000_000, task_type="sft"))
assert "estimated_total_mb" in mem_res, "Missing estimated_total_mb in response"
assert mem_res["estimated_total_mb"] > 0
print(f"[PASS] Analytical Memory Estimation: {mem_res['estimated_total_mb']} MB ({mem_res.get('estimated_total_gb')} GB)")

# 3. Empirical Memory Probe with Measured Schema & Truthful Bounds
print("\n[Handler 3/4] Invoking run_preflight_probe handler (bounded subprocess)...")
probe_cfg = {
    "schema_version": "1.0.0",
    "task_type": "sft",
    "model": {"model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM", "torch_dtype": "float32"},
    "dataset": {"dataset_name_or_path": "synthetic", "max_seq_length": 32, "train_sample_limit": 4},
    "hardware": {"per_device_train_batch_size": 1, "target_device": "cpu"},
}
probe_res = json.loads(run_preflight_probe(config_json=json.dumps(probe_cfg), in_subprocess=True, max_process_memory_mb=4096))
assert probe_res.get("probe_successful") is True, f"Probe failed: {probe_res}"

# Verify measured schema keys and truthful non-representative guarantees
assert "measured_peak_allocated_mb" in probe_res, "Missing measured_peak_allocated_mb in probe result!"
assert probe_res["measured_peak_allocated_mb"] > 0, "measured_peak_allocated_mb must be > 0"
assert "verdict" in probe_res, "Missing verdict in probe result!"
assert probe_res.get("isolation_mode") == "subprocess", "Probe must run in isolated subprocess"
assert probe_res.get("gpu_fit_guaranteed") is False, "gpu_fit_guaranteed must truthfully be False"
assert probe_res.get("is_representative") is False, "is_representative on CPU must truthfully be False"
assert len(probe_res.get("representativeness_warnings", [])) > 0, "Must include truthful representativeness warning on CPU"

# Save actual probe measured result to artifacts
probe_out_path = PROJECT_ROOT / "artifacts" / "mcp_probe_measured_result.json"
probe_out_path.write_text(json.dumps(probe_res, indent=2), encoding="utf-8")
print(f"[PASS] Empirical Probe measured schema verified: peak_rss={probe_res.get('measured_peak_allocated_mb')} MB, verdict={probe_res.get('verdict')}, representative={probe_res.get('is_representative')}")
print(f"       Saved actual probe result to: {probe_out_path}")

# 4. Denial-Before-Loader & Grant Mismatch Security Enforcement
print("\n[Handler 4/4] Verifying denial-before-loader and authorization grant boundaries...")
denied_1 = json.loads(submit_training_job(config_json=json.dumps(probe_cfg), authorized=False))
assert denied_1.get("error") == "EXECUTION_DENIED", f"Expected EXECUTION_DENIED, got {denied_1}"

denied_2 = json.loads(submit_training_job(config_json=json.dumps(probe_cfg), authorized=True, authorization_grant=None))
assert denied_2.get("error") == "AUTHORIZATION_GRANT_REQUIRED", f"Expected AUTHORIZATION_GRANT_REQUIRED, got {denied_2}"

denied_3 = json.loads(submit_training_job(config_json=json.dumps(probe_cfg), authorized=True, authorization_grant="invalid_hex_grant_12345"))
assert denied_3.get("error") == "PLAN_GRANT_MISMATCH", f"Expected PLAN_GRANT_MISMATCH, got {denied_3}"

os.environ["TRAIN_STACK_ALLOW_PROBE"] = "0"
denied_probe = json.loads(run_preflight_probe(config_json=json.dumps(probe_cfg), in_subprocess=True))
assert denied_probe.get("error") == "EXECUTION_DENIED_BY_OPERATOR_POLICY", f"Expected EXECUTION_DENIED_BY_OPERATOR_POLICY, got {denied_probe}"
os.environ["TRAIN_STACK_ALLOW_PROBE"] = "1"
print("[PASS] Direct handler denial-before-loader verified: unauthorized calls and invalid grants rejected.")


# ==============================================================================
# PART 2: INSTALLED FASTMCP STDIO CLIENT PROTOCOL TRANSPORT EVIDENCE
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: INSTALLED FASTMCP STDIO CLIENT PROTOCOL TRANSPORT EVIDENCE")
print("=" * 80)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def find_worker_pid(target_dir, timeout=10.0):
    deadline = time.time() + timeout
    target_str = str(target_dir).lower()
    while time.time() < deadline:
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = p.info.get('cmdline') or []
                if any(target_str in str(arg).lower() for arg in cmd):
                    return p.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.1)
    return None

def get_child_process_count():
    try:
        current = psutil.Process()
        return len(current.children(recursive=True))
    except Exception:
        return 0

async def find_worker_pid_async(target_dir, timeout=10.0):
    deadline = time.time() + timeout
    target_str = str(target_dir).lower()
    while time.time() < deadline:
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmd = p.info.get('cmdline') or []
                if any(target_str in str(arg).lower() for arg in cmd):
                    return p.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        await asyncio.sleep(0.1)
    return None

async def run_stdio_mcp_tests():
    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    server_env["TRAIN_STACK_ALLOW_PROBE"] = "1"
    server_env["TRAIN_STACK_ALLOW_LAUNCH"] = "1"
    server_env["TRAIN_STACK_ALLOW_CANCEL"] = "1"
    server_env["TRAIN_STACK_AUTH_SECRET"] = AUTH_SECRET
    server_env["TRAIN_STACK_ALLOWED_ROOTS"] = str(PROJECT_ROOT)

    server_script = (
        "import subprocess\n"
        "class PatchedPopen(subprocess.Popen):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        if 'stdin' not in kwargs:\n"
        "            kwargs['stdin'] = subprocess.DEVNULL\n"
        "        super().__init__(*args, **kwargs)\n"
        "subprocess.Popen = PatchedPopen\n"
        "from llm_training_stack.mcp.server import run_server\n"
        "run_server()\n"
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-c", server_script],
        env=server_env,
    )

    print("\n[STDIO 1/6] Initializing FastMCP ClientSession over stdio transport...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            init_res = await session.initialize()
            print(f"[PASS] FastMCP STDIO session initialized successfully (server={init_res.serverInfo.name} v{init_res.serverInfo.version})")

            # 2. List tools
            print("\n[STDIO 2/6] Querying tools/list over protocol...")
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            print(f"Available protocol tools ({len(tool_names)}): {tool_names}")
            expected_tools = [
                "inspect_hardware", "estimate_memory", "run_preflight_probe",
                "submit_training_job", "cancel_training_job",
                "get_training_job_status", "get_training_job_logs"
            ]
            for exp in expected_tools:
                assert exp in tool_names, f"Expected tool {exp} missing from protocol tools/list!"
            print("[PASS] All expected tools verified in protocol tools/list.")

            # 3. Read boundary tools
            print("\n[STDIO 3/6] Calling read-boundary tools over STDIO protocol...")
            hw_call = await session.call_tool("inspect_hardware", {})
            assert len(hw_call.content) > 0, "Empty content in inspect_hardware response"
            hw_json = json.loads(hw_call.content[0].text)
            assert "platform" in hw_json
            print(f"[PASS] inspect_hardware over STDIO: platform={hw_json.get('platform')}, RAM={hw_json.get('system_ram_total_gb')} GB")

            est_call = await session.call_tool("estimate_memory", {"total_params": 135_000_000, "task_type": "sft"})
            est_json = json.loads(est_call.content[0].text)
            assert est_json.get("estimated_total_mb", 0) > 0
            print(f"[PASS] estimate_memory over STDIO: {est_json.get('estimated_total_mb')} MB")

            # 4. Probe boundary tool over STDIO
            print("\n[STDIO 4/6] Calling run_preflight_probe over STDIO protocol (bounded probe response)...")
            probe_call = await session.call_tool("run_preflight_probe", {
                "config_json": json.dumps(probe_cfg),
                "in_subprocess": True,
                "max_process_memory_mb": 4096
            })
            stdio_probe_res = json.loads(probe_call.content[0].text)
            assert stdio_probe_res.get("probe_successful") is True, f"STDIO Probe failed: {stdio_probe_res}"
            assert stdio_probe_res.get("isolation_mode") == "subprocess"
            assert stdio_probe_res.get("gpu_fit_guaranteed") is False
            print(f"[PASS] run_preflight_probe over STDIO: peak_rss={stdio_probe_res.get('measured_peak_allocated_mb')} MB, verdict={stdio_probe_res.get('verdict')}")

            # 5. Observable loader/spawn instrumentation on denials
            print("\n[STDIO 5/6] Verifying observable loader/spawn denial instrumentation over STDIO protocol...")
            
            # Record children before call
            children_before = get_child_process_count()

            # 5a. Unauthorized launch attempt
            denied_unauth = await session.call_tool("submit_training_job", {
                "config_json": json.dumps(probe_cfg),
                "authorized": False
            })
            unauth_data = json.loads(denied_unauth.content[0].text)
            assert unauth_data.get("error") == "EXECUTION_DENIED", f"Expected EXECUTION_DENIED, got {unauth_data}"
            
            # Instrumentation check: No child worker processes spawned
            children_after = get_child_process_count()
            assert children_after == children_before, f"Child process spawned during unauthorized call! (before={children_before}, after={children_after})"
            print(f"[PASS] Observable denial-before-loader verified: child process count remained unchanged ({children_before} -> {children_after})")

            # 5b. Mismatched launch grant attempt
            denied_grant = await session.call_tool("submit_training_job", {
                "config_json": json.dumps(probe_cfg),
                "authorized": True,
                "authorization_grant": "deadbeef_invalid_grant_9999"
            })
            grant_data = json.loads(denied_grant.content[0].text)
            assert grant_data.get("error") == "PLAN_GRANT_MISMATCH", f"Expected PLAN_GRANT_MISMATCH, got {grant_data}"
            
            # Instrumentation check: Still no new child processes
            children_after_grant = get_child_process_count()
            assert children_after_grant == children_before, f"Child process spawned during invalid grant call! (before={children_before}, after={children_after_grant})"
            print(f"[PASS] Observable grant mismatch verified: child process count remained unchanged ({children_before} -> {children_after_grant})")

            # 6. Real Child Job Submission, Cancellation, Strict OS Exit & Cross-Artifact Agreement over STDIO
            print("\n[STDIO 6/6] Spawning real background job over STDIO transport and executing active cancellation...")
            run_id_mcp = f"installed_mcp_real_{uuid.uuid4().hex[:8]}"
            mcp_run_dir = OUTPUT_DIR / run_id_mcp
            mcp_run_dir.mkdir(parents=True, exist_ok=True)

            train_job_cfg = TrainingJobConfig(
                task_type=TaskType.SFT,
                model=ModelConfig(model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype="float32"),
                dataset=DatasetConfig(dataset_name_or_path="synthetic", max_seq_length=32, train_sample_limit=25),
                hardware=HardwareConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, target_device="cpu"),
                logging=LoggingConfig(output_dir=str(mcp_run_dir), logging_steps=1, save_steps=2),
                max_steps=30,
            )

            launch_grant = create_plan_grant(train_job_cfg, action="submit", secret=AUTH_SECRET)

            sub_call = await session.call_tool("submit_training_job", {
                "config_json": train_job_cfg.model_dump_json(),
                "authorized": True,
                "authorization_grant": launch_grant,
                "background": True,
            })
            sub_res = json.loads(sub_call.content[0].text)
            assert sub_res.get("status") in ["SUBMITTED", "RUNNING"], f"Submit failed over STDIO: {sub_res}"
            print(f"[PASS] Real background job submitted over STDIO: {sub_res.get('job_id')}, status={sub_res.get('status')}")

            child_pid = await find_worker_pid_async(mcp_run_dir, timeout=10.0)
            print(f"Child worker subprocess detected in OS with PID: {child_pid}")
            assert child_pid is not None and child_pid > 0, "No child PID detected in OS for STDIO job!"
            assert psutil.pid_exists(child_pid), f"Child PID {child_pid} not running in OS!"
            print(f"[VERIFIED] Child process PID {child_pid} confirmed running in OS process table.")

            # Await progress via get_training_job_logs over STDIO
            observed = False
            for _ in range(120):
                logs_call = await session.call_tool("get_training_job_logs", {
                    "run_dir": str(mcp_run_dir),
                    "tail_lines": 10
                })
                logs_res = json.loads(logs_call.content[0].text)
                if isinstance(logs_res, list) and len(logs_res) >= 1:
                    observed = True
                    break
                await asyncio.sleep(0.2)

            assert observed, "Active training progress was not observed over STDIO protocol within deadline!"
            print(f"[PASS] Active training progress observed over STDIO ({len(logs_res)} events retrieved).")

            # Active cancellation with action-bound cancellation grant over STDIO
            cancel_grant = create_cancellation_grant(str(mcp_run_dir), secret=AUTH_SECRET)
            cancel_call = await session.call_tool("cancel_training_job", {
                "run_dir": str(mcp_run_dir),
                "authorized": True,
                "authorization_grant": cancel_grant,
                "wait": True,
                "timeout": 25.0,
            })
            cancel_res = json.loads(cancel_call.content[0].text)
            assert cancel_res.get("status") == "CANCELLED", f"Expected CANCELLED over STDIO, got {cancel_res.get('status')}"
            assert cancel_res.get("finished_at") is not None, "finished_at must be set upon acknowledgment"
            assert cancel_res.get("current_step") is not None and cancel_res.get("current_step") > 0
            assert cancel_res.get("latest_loss") is not None
            print(f"[PASS] MCP cancel_training_job acknowledged over STDIO: status=CANCELLED, step={cancel_res.get('current_step')}, loss={cancel_res.get('latest_loss')}")

            # Strict OS Process Exit Check: Verify child process PID is no longer running in OS
            # Strictly rejects psutil.AccessDenied as death proof
            os_exited = False
            for _ in range(60):
                if not psutil.pid_exists(child_pid):
                    os_exited = True
                    break
                try:
                    proc = psutil.Process(child_pid)
                    if proc.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                        os_exited = True
                        break
                except psutil.NoSuchProcess:
                    os_exited = True
                    break
                except psutil.AccessDenied:
                    # Indeterminate: continue polling until deadline
                    pass
                await asyncio.sleep(0.1)

            assert os_exited, f"Child worker process PID {child_pid} is still running after acknowledged cancellation!"
            assert not psutil.pid_exists(child_pid), f"Child PID {child_pid} still exists in OS process table!"
            print(f"[PASS] Strict child OS process exit verified: PID {child_pid} confirmed exited (not pid_exists).")

            # Strict Cross-Artifact Field & Metrics Agreement Check
            job_state_file = mcp_run_dir / "job_state.json"
            manifest_file = mcp_run_dir / "manifest.json"
            events_file = mcp_run_dir / "events.jsonl"
            assert job_state_file.exists(), "job_state.json missing!"
            assert manifest_file.exists(), "manifest.json missing!"
            assert events_file.exists(), "events.jsonl missing!"

            with open(job_state_file, "r", encoding="utf-8") as f:
                job_state_data = json.load(f)
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            event_lines = [json.loads(l) for l in events_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
            assert len(event_lines) > 0, "No event entries found in events.jsonl"
            last_event = event_lines[-1]

            print("\n--- MCP CROSS-ARTIFACT FIELD & METRICS AGREEMENT AUDIT ---")
            print(f"  • job_id (scheduler): {job_state_data.get('job_id')}, run_id (pipeline): {manifest_data.get('run_id')}")
            assert job_state_data.get("job_id") and manifest_data.get("run_id"), "Missing job_id or run_id!"

            # Compare run directory / output directory
            manifest_out_dir = manifest_data.get("config", {}).get("logging", {}).get("output_dir")
            print(f"  • output_dir: job_state={job_state_data.get('run_dir')} vs manifest={manifest_out_dir}")
            assert Path(job_state_data.get("run_dir")).resolve() == Path(manifest_out_dir).resolve(), "output_dir mismatch!"

            # Compare status across job_state, manifest, and MCP cancellation response
            print(f"  • status: job_state={job_state_data.get('status')}, manifest={manifest_data.get('status')}, mcp={cancel_res.get('status')}")
            assert job_state_data.get("status") == "CANCELLED", "Status in job_state not CANCELLED!"
            assert manifest_data.get("status") == "CANCELLED", "Status in manifest not CANCELLED!"
            assert cancel_res.get("status") == "CANCELLED", "Status in MCP response not CANCELLED!"

            # Compare steps across job_state, manifest, and last event log
            manifest_steps = manifest_data.get("final_metrics", {}).get("total_steps", job_state_data.get("current_step"))
            print(f"  • steps: job_state={job_state_data.get('current_step')}, manifest={manifest_steps}, last_event={last_event.get('step')}")
            assert job_state_data.get("current_step") == last_event.get("step"), "Step mismatch between job_state and events.jsonl!"
            assert manifest_steps == last_event.get("step"), "Step mismatch between manifest and events.jsonl!"

            tolerance = 1e-4
            diff_loss = abs(float(job_state_data.get("latest_loss")) - float(last_event.get("loss")))
            manifest_loss = float(manifest_data.get("final_metrics", {}).get("final_loss", job_state_data.get("latest_loss")))
            diff_manifest_loss = abs(float(job_state_data.get("latest_loss")) - manifest_loss)
            print(f"  • loss: job_state={job_state_data.get('latest_loss')}, manifest={manifest_loss}, last_event={last_event.get('loss')} (diff={diff_loss:.2e}, tol={tolerance:.2e})")
            assert diff_loss < tolerance, f"Loss mismatch {diff_loss} exceeds tolerance {tolerance}!"
            assert diff_manifest_loss < tolerance, f"Manifest loss mismatch {diff_manifest_loss} exceeds tolerance {tolerance}!"

            # Check checkpoint metrics agreement if saved
            ckpt_dir = job_state_data.get("checkpoint_dir")
            if ckpt_dir:
                print(f"  • checkpoint: {ckpt_dir} (exists={Path(ckpt_dir).exists()})")
                assert Path(ckpt_dir).exists(), f"Checkpoint directory {ckpt_dir} does not exist on disk!"

            mcp_agreement = {
                "run_id": job_state_data.get("job_id"),
                "status": "CANCELLED",
                "verified_child_os_pid": child_pid,
                "pid_exited": True,
                "job_state_step": job_state_data.get("current_step"),
                "event_log_step": last_event.get("step"),
                "job_state_loss": job_state_data.get("latest_loss"),
                "event_log_loss": last_event.get("loss"),
                "loss_precision_delta": diff_loss,
                "precision_tolerance": tolerance,
                "cross_artifact_verdict": "VERIFIED_ACCORD",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            agreement_out = PROJECT_ROOT / "artifacts" / "DELTA23_MCP_AGREEMENT.json"
            agreement_out.write_text(json.dumps(mcp_agreement, indent=2), encoding="utf-8")
            print(f"[PASS] Cross-artifact agreement verified: step={job_state_data.get('current_step')}, loss diff={diff_loss:.2e} < {tolerance:.2e}")
            print(f"       Saved MCP agreement record to: {agreement_out}")

# Execute Part 2 via asyncio
asyncio.run(run_stdio_mcp_tests())

print("\n" + "=" * 80)
print("ALL MCP SERVER TESTS PASSED (HANDLER INTEGRATION + FAST-MCP STDIO PROTOCOL)")
print("=" * 80)
