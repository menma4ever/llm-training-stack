"""Master Portable Candidate Verification Orchestration Harness (DELTA 23).
Executes all candidate verification stages in the dedicated installed virtualenv:
1. pip check
2. Module provenance and isolation audit
3. CLI entry point tests (help, inspect, preflight)
4. Pinned SmolLM-135M adaptation verification
5. Targeted lifecycle management and child OS process exit verification
6. Targeted MCP server tools with measured probe schema, denial-before-loader, FastMCP STDIO transport, real child cancellation & strict OS exit
7. Strict RunComparator & incompatible negative cases
8. Comprehensive Installed CLI Workflow Equivalence:
   - CLI inspect & preflight
   - CLI background submission
   - Active CLI cancellation during in-progress training (verified transition to CANCELLED)
   - CLI status & structured logs
   - CLI cancellation on terminal job
   - CLI held-out evaluation & run comparison
Non-destructive: preserves historical runs; outputs durable immutable DELTA 23 logs to artifacts/.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import yaml

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Master portable candidate verification orchestration harness")
parser.add_argument("--project-root", type=str, default=None, help="Explicit path to project root")
parser.add_argument("--wheel-path", type=str, default=None, help="Explicit path to candidate wheel")
parser.add_argument("--venv-dir", type=str, default=None, help="Explicit path to candidate virtual environment")
parser.add_argument("--evidence-log", type=str, default=None, help="Explicit path to master evidence log")
parser.add_argument("--cli-log", type=str, default=None, help="Explicit path to CLI workflow evidence log")
args, _ = parser.parse_known_args()

if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
elif os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

VERIF_DIR = SCRIPT_DIR

if args.wheel_path:
    WHEEL_PATH = Path(args.wheel_path).resolve()
else:
    WHEEL_PATH = PROJECT_ROOT / "artifacts" / "llm_training_stack-0.1.0-py3-none-any.whl"
    if not WHEEL_PATH.exists():
        WHEEL_PATH = PROJECT_ROOT / "shared" / "llm_training_stack-0.1.0-py3-none-any.whl"

VENV_DIR = Path(args.venv_dir).resolve() if args.venv_dir else (PROJECT_ROOT / "artifacts" / "installed_candidate_venv")
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
CLI_EXE = VENV_DIR / "Scripts" / "train-stack.exe"

EVIDENCE_LOG = Path(args.evidence_log).resolve() if args.evidence_log else (PROJECT_ROOT / "artifacts" / "DELTA23_FINAL_INSTALLED_CANDIDATE_EVIDENCE.log")
CLI_EVIDENCE_LOG = Path(args.cli_log).resolve() if args.cli_log else (PROJECT_ROOT / "artifacts" / "DELTA23_INSTALLED_CLI_WORKFLOW_EVIDENCE.log")

log_lines = []
cli_log_lines = []

def log(msg="", for_cli=False):
    print(msg)
    log_lines.append(str(msg))
    if for_cli:
        cli_log_lines.append(str(msg))

def run_cmd(cmd, cwd=None, env=None, check=True, for_cli=False):
    log(f"\n[EXEC]: {cmd}", for_cli=for_cli)
    t0 = time.time()
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or str(PROJECT_ROOT),
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.time() - t0
    if res.stdout:
        log(f"[STDOUT]:\n{res.stdout.strip()}", for_cli=for_cli)
    if res.stderr:
        log(f"[STDERR]:\n{res.stderr.strip()}", for_cli=for_cli)
    log(f"[EXIT CODE]: {res.returncode} (took {duration:.2f}s)", for_cli=for_cli)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res

log("=" * 80)
log("CEO DELTA 23: PORTABLE CANDIDATE HARNESS & COMPLETE EVIDENCE EXECUTION")
log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
log(f"Project Root: {PROJECT_ROOT}")
log(f"Harness Directory: {VERIF_DIR}")
log(f"Candidate Wheel: {WHEEL_PATH}")
assert WHEEL_PATH.exists(), f"Wheel not found at {WHEEL_PATH}"

wheel_bytes = WHEEL_PATH.read_bytes()
wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()
log(f"Wheel Size: {len(wheel_bytes)} bytes")
log(f"Wheel SHA256: {wheel_sha}")
assert len(wheel_bytes) == 83637, f"Unexpected wheel size: {len(wheel_bytes)}"
assert wheel_sha == "dbaabab553387ccad26f8f724446de3c6746ed8039b11a30eb8d7cce1d0865a1"
log("[PASS] Candidate wheel identity verified: 83,637 bytes, SHA256 dbaabab5...")
log("=" * 80)

# Verify virtual environment
assert VENV_PY.exists(), f"Candidate virtualenv Python not found at {VENV_PY}"
assert CLI_EXE.exists(), f"Candidate CLI executable not found at {CLI_EXE}"

# 1. pip check in installed candidate virtualenv
log("\n--- STEP 1: PIP CHECK IN ISOLATED VIRTUALENV ---")
run_cmd(f'"{VENV_PY}" -m pip check')

# 2. Module Provenance and Isolation Audit
log("\n--- STEP 2: MODULE PROVENANCE & ISOLATION AUDIT ---")
run_cmd(f'"{VENV_PY}" "{VERIF_DIR / "audit_provenance_candidate.py"}"')

# 3. CLI Entry Points Validation
log("\n--- STEP 3: CLI ENTRY POINTS BASIC VALIDATION ---")
run_cmd(f'"{CLI_EXE}" --help')
run_cmd(f'"{CLI_EXE}" inspect')
run_cmd(f'"{CLI_EXE}" preflight --config "shared/llm-training-stack/examples/training_config.yaml"')

# 4. Pinned SmolLM-135M Adaptation Verification
log("\n--- STEP 4: PINNED SMOLLM-135M ADAPTATION & EVALUATION ---")
run_cmd(f'"{VENV_PY}" "{VERIF_DIR / "run_adaptation_in_candidate.py"}"')

# 5. Targeted Lifecycle Management with Child Process OS Exit Check
log("\n--- STEP 5: TARGETED LIFECYCLE MANAGEMENT & STRICT OS EXIT CHECK ---")
run_cmd(f'"{VENV_PY}" "{VERIF_DIR / "run_lifecycle_in_candidate.py"}" --project-root "{PROJECT_ROOT}"')

# 6. Targeted MCP Server Validation (Handler Integration + FastMCP STDIO Protocol)
log("\n--- STEP 6: TARGETED MCP SERVER TOOLS & FASTMCP STDIO PROTOCOL ---")
run_cmd(f'"{VENV_PY}" "{VERIF_DIR / "run_mcp_in_candidate.py"}" --project-root "{PROJECT_ROOT}"')

# 7. Strict RunComparator & Incompatible Negative Cases
log("\n--- STEP 7: STRICT RUN COMPARATOR & INCOMPATIBLE NEGATIVE CASES ---")
run_cmd(f'"{VENV_PY}" "{VERIF_DIR / "test_strict_comparator.py"}"')

# 8. Comprehensive Installed CLI Workflow Equivalence (inspect, preflight, submit, active cancel, status, logs, terminal cancel, eval, compare)
log("\n--- STEP 8: INSTALLED CLI WORKFLOW EQUIVALENCE AUDIT ---", for_cli=True)

# 8a. CLI Inspect
log("\n[CLI 1/8] train-stack inspect", for_cli=True)
run_cmd(f'"{CLI_EXE}" inspect', for_cli=True)

# 8b. CLI Preflight
log("\n[CLI 2/8] train-stack preflight", for_cli=True)
run_cmd(f'"{CLI_EXE}" preflight --config "shared/llm-training-stack/examples/training_config.yaml"', for_cli=True)

# 8c. Prepare Active Cancellation Test Configuration
cli_active_cancel_dir = PROJECT_ROOT / "artifacts" / "candidate_runs" / "cli_active_cancel_run"
cli_active_cancel_dir.mkdir(parents=True, exist_ok=True)

cancel_config_dict = {
    "schema_version": "1.0.0",
    "task_type": "lora",
    "model": {
        "model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "trust_remote_code": False,
        "torch_dtype": "float32",
    },
    "dataset": {
        "dataset_name_or_path": "synthetic",
        "max_seq_length": 64,
        "train_sample_limit": 500,
    },
    "hardware": {
        "target_device": "cpu",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
    },
    "logging": {
        "output_dir": str(cli_active_cancel_dir),
        "logging_steps": 1,
        "save_steps": 5,
    },
    "max_steps": 150,
}
active_cancel_yaml_path = PROJECT_ROOT / "artifacts" / "cli_active_cancel_config.yaml"
active_cancel_yaml_path.write_text(yaml.dump(cancel_config_dict), encoding="utf-8")

# 8d. CLI Submit Background Job
log("\n[CLI 3/8] train-stack submit (active background job)", for_cli=True)
sub_res = run_cmd(f'"{CLI_EXE}" submit --config "{active_cancel_yaml_path}" --background', for_cli=True)

import re
job_id_match = re.search(r"ID=([a-zA-Z0-9_]+)", sub_res.stdout)
assert job_id_match, f"Failed to extract Job ID from submit output: {sub_res.stdout}"
job_target = job_id_match.group(1)
log(f"Targeting submitted job by ID: {job_target}", for_cli=True)

# 8e. Await Active Execution
log(f"\n[CLI 4/8] Polling train-stack status until actively RUNNING...", for_cli=True)
is_running = False
for _ in range(60):
    st_res = run_cmd(f'"{CLI_EXE}" status "{job_target}"', check=False, for_cli=False)
    if "RUNNING" in st_res.stdout:
        is_running = True
        log(f"Job {job_target} confirmed actively RUNNING in OS table.", for_cli=True)
        break
    time.sleep(0.2)

assert is_running, f"Job {job_target} did not enter RUNNING status within deadline!"

# 8f. Active CLI Cancellation: train-stack cancel WHILE in progress
log(f"\n[CLI 5/8] train-stack cancel (ACTIVE IN-PROGRESS CANCELLATION)", for_cli=True)
cancel_res = run_cmd(f'"{CLI_EXE}" cancel "{job_target}"', for_cli=True)
assert "cancelled. Status: CANCELLED" in cancel_res.stdout or "cancellation requested" in cancel_res.stdout, f"Expected active cancellation message, got: {cancel_res.stdout}"

# Verify status reports CANCELLED
log(f"\n[CLI 6/8] train-stack status & logs post-cancellation", for_cli=True)
for _ in range(30):
    status_post = run_cmd(f'"{CLI_EXE}" status "{job_target}"', check=False, for_cli=False)
    if "CANCELLED" in status_post.stdout:
        break
    time.sleep(0.5)

status_post = run_cmd(f'"{CLI_EXE}" status "{job_target}"', for_cli=True)
assert "CANCELLED" in status_post.stdout, f"Expected CANCELLED status, got: {status_post.stdout}"
run_cmd(f'"{CLI_EXE}" logs "{job_target}"', for_cli=True)

# 8g. Terminal-Job CLI Cancellation Check (Demonstrating accurate terminal-job message)
log(f"\n[CLI 7/8] train-stack cancel on already-terminated job", for_cli=True)
cancel_terminal = run_cmd(f'"{CLI_EXE}" cancel "{job_target}"', for_cli=True)
assert "had already completed prior to cancellation" in cancel_terminal.stdout or "cancelled. Status: CANCELLED" in cancel_terminal.stdout, f"Unexpected output: {cancel_terminal.stdout}"

# 8h. CLI Eval & Compare
cli_eval_report = PROJECT_ROOT / "artifacts" / "cli_eval_report.json"
log("\n[CLI 8/8] train-stack eval & compare", for_cli=True)
run_cmd(
    f'"{CLI_EXE}" eval --model "hf-internal-testing/tiny-random-LlamaForCausalLM" '
    f'--dataset "artifacts/real_heldout_dataset.jsonl" --max-seq-length 32 --sample-limit 3 --output-report "{cli_eval_report}"',
    for_cli=True
)
run_cmd(
    f'"{CLI_EXE}" compare "artifacts/installed_candidate_baseline" "artifacts/installed_candidate_run"',
    for_cli=True
)

log("\n" + "=" * 80)
log("ALL CANDIDATE HARNESS VERIFICATION GATES PASSED (100% SUCCESS)")
log("=" * 80)

# Save Master Evidence Log (DELTA 23)
EVIDENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
EVIDENCE_LOG.write_text("\n".join(log_lines), encoding="utf-8")
print(f"\nDurable Master Evidence Log saved to: {EVIDENCE_LOG} ({EVIDENCE_LOG.stat().st_size} bytes)")

# Save Candidate Orchestration Harness Log
HARNESS_LOG = PROJECT_ROOT / "artifacts" / "DELTA23_CANDIDATE_ORCHESTRATION_HARNESS.log"
HARNESS_LOG.write_text("\n".join(log_lines), encoding="utf-8")
print(f"Durable Candidate Orchestration Harness Log saved to: {HARNESS_LOG} ({HARNESS_LOG.stat().st_size} bytes)")

# Save CLI Workflow Evidence Log
CLI_EVIDENCE_LOG.write_text("\n".join(cli_log_lines), encoding="utf-8")
print(f"Durable CLI Workflow Evidence Log saved to: {CLI_EVIDENCE_LOG} ({CLI_EVIDENCE_LOG.stat().st_size} bytes)")

# Keep legacy names synchronized
legacy_evidence = PROJECT_ROOT / "artifacts" / "FINAL_INSTALLED_CANDIDATE_EVIDENCE.log"
legacy_evidence.write_text("\n".join(log_lines), encoding="utf-8")
legacy_cli = PROJECT_ROOT / "artifacts" / "INSTALLED_CLI_WORKFLOW_EVIDENCE.log"
legacy_cli.write_text("\n".join(cli_log_lines), encoding="utf-8")

# Sync to shared/
shared_evidence = PROJECT_ROOT / "shared" / "DELTA23_FINAL_INSTALLED_CANDIDATE_EVIDENCE.log"
shared_harness = PROJECT_ROOT / "shared" / "DELTA23_CANDIDATE_ORCHESTRATION_HARNESS.log"
shared_cli = PROJECT_ROOT / "shared" / "DELTA23_INSTALLED_CLI_WORKFLOW_EVIDENCE.log"
shared_evidence.write_text(EVIDENCE_LOG.read_text(encoding="utf-8"), encoding="utf-8")
shared_harness.write_text(HARNESS_LOG.read_text(encoding="utf-8"), encoding="utf-8")
shared_cli.write_text(CLI_EVIDENCE_LOG.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Synchronized logs to shared/ directory.")
