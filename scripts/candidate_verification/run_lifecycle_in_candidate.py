"""Targeted lifecycle management and cancellation validation in installed candidate virtualenv.
Validates:
1. Unacknowledged cancellation timeout retains RUNNING and finished_at=None on disk.
2. Active child worker subprocess cancellation acknowledges CANCELLED, records step/loss metrics,
   and verifies child process OS exit (PID termination check).
3. Non-destructive unique run directories; never uses rmtree.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
import psutil

import argparse

# Project root resolution: explicit CLI argument -> env var -> auto-detection
parser = argparse.ArgumentParser(description="Targeted lifecycle management validation in candidate virtualenv")
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

from llm_training_stack.config.schema import (
    TrainingJobConfig, TaskType, ModelConfig, DatasetConfig,
    HardwareConfig, LoggingConfig
)
from llm_training_stack.lifecycle.manager import LifecycleManager, JobStatus, JobRecord

print("=" * 80)
print("TARGETED LIFECYCLE MANAGEMENT VALIDATION IN INSTALLED CANDIDATE")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)

# 1. Unacknowledged cancellation retains RUNNING and finished_at unset
run_id_unack = f"installed_lifecycle_unacked_{uuid.uuid4().hex[:8]}"
tmp_dir = PROJECT_ROOT / "artifacts" / "candidate_runs" / run_id_unack
tmp_dir.mkdir(parents=True, exist_ok=True)

rec = JobRecord(
    job_id=run_id_unack,
    run_dir=str(tmp_dir),
    task_type=TaskType.SFT,
    config={"task_type": "sft"},
    status=JobStatus.RUNNING,
    created_at=datetime.now(timezone.utc).isoformat(),
)
rec.save(tmp_dir)

print(f"\n[Test 1/2] Calling cancel_job on RUNNING job with timeout and no child worker ({tmp_dir})...")
res = LifecycleManager.cancel_job(tmp_dir, wait=True, timeout=0.2)
assert (tmp_dir / "cancel.token").exists(), "cancel.token was not created!"
assert res.status == JobStatus.RUNNING, f"Status changed to {res.status}, expected RUNNING!"
assert res.finished_at is None, f"finished_at set prematurely to {res.finished_at}!"
disk_rec = LifecycleManager.get_job_status(tmp_dir)
assert disk_rec.status == JobStatus.RUNNING, "Disk status was overwritten!"
assert disk_rec.finished_at is None, "Disk finished_at was overwritten!"
print("[PASS] Unacknowledged cancellation correctly retained RUNNING and finished_at unset on disk.")

# 2. Active child worker subprocess cancellation with verified OS process exit
run_id_active = f"installed_lifecycle_active_{uuid.uuid4().hex[:8]}"
active_dir = PROJECT_ROOT / "artifacts" / "candidate_runs" / run_id_active
active_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[Test 2/2] Spawning active worker subprocess and verifying clean cancellation acknowledgment & OS exit ({active_dir})...")
cfg = TrainingJobConfig(
    task_type=TaskType.SFT,
    model=ModelConfig(model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype="float32"),
    dataset=DatasetConfig(dataset_name_or_path="synthetic", max_seq_length=32, train_sample_limit=20),
    hardware=HardwareConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, target_device="cpu"),
    logging=LoggingConfig(output_dir=str(active_dir), logging_steps=1, save_steps=2),
    max_steps=25,
)

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

sub_rec = LifecycleManager.submit_job(cfg, run_in_background=True)
assert sub_rec.status == JobStatus.SUBMITTED

child_pid = find_worker_pid(active_dir, timeout=10.0)
print(f"Child worker subprocess detected in OS with PID: {child_pid}")
assert child_pid is not None and child_pid > 0, "No child worker process detected in OS for active job!"

# Verify child process exists in OS
assert psutil.pid_exists(child_pid), f"Child process PID {child_pid} does not exist in OS table!"
print(f"[VERIFIED] Child process PID {child_pid} confirmed running in OS process table.")

# Wait for active progress
events_file = active_dir / "events.jsonl"
observed = False
for _ in range(120):
    if events_file.exists():
        logs = LifecycleManager.get_job_logs(active_dir)
        if len(logs) >= 1:
            observed = True
            break
    time.sleep(0.2)

assert observed, "Active training progress was not observed within deadline!"
print(f"Active training progress observed ({len(logs)} event log entries recorded).")

# Cancel active job
cancel_res = LifecycleManager.cancel_job(active_dir, wait=True, timeout=25.0)
assert (active_dir / "cancel.token").exists()
assert cancel_res.status == JobStatus.CANCELLED, f"Expected CANCELLED, got {cancel_res.status}"
assert cancel_res.finished_at is not None, "finished_at must be set upon child acknowledgment"
assert cancel_res.current_step is not None and cancel_res.current_step > 0
assert cancel_res.latest_loss is not None
assert (active_dir / "manifest.json").exists()

# Explicit OS Process Exit Check: Verify child process PID is no longer running in OS
# Strict policy: psutil.AccessDenied is indeterminate and must NOT pass a death assertion
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
        # AccessDenied is indeterminate; continue polling until confirmed dead or deadline
        pass
    time.sleep(0.1)

assert os_exited, f"Child worker process PID {child_pid} is still running after acknowledged cancellation!"
assert not psutil.pid_exists(child_pid), f"Child PID {child_pid} still exists in OS process table!"
print(f"[PASS] Child worker process PID {child_pid} confirmed exited in OS process table (strict NoSuchProcess / not pid_exists).")
print(f"[PASS] Active child worker cancellation acknowledged: status=CANCELLED, step={cancel_res.current_step}, loss={cancel_res.latest_loss:.4f}, finished_at={cancel_res.finished_at}")

# Cross-Artifact Field & Metrics Agreement Verification
job_state_file = active_dir / "job_state.json"
manifest_file = active_dir / "manifest.json"
assert job_state_file.exists(), f"job_state.json missing at {job_state_file}"
assert manifest_file.exists(), f"manifest.json missing at {manifest_file}"

with open(job_state_file, "r", encoding="utf-8") as f:
    job_state_data = json.load(f)
with open(manifest_file, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

event_lines = [json.loads(l) for l in events_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
assert len(event_lines) > 0, "No event records found in events.jsonl"
last_event = event_lines[-1]

print("\n--- CROSS-ARTIFACT FIELD & METRICS AGREEMENT AUDIT ---")
print(f"  • job_id (scheduler): {job_state_data.get('job_id')}, run_id (pipeline): {manifest_data.get('run_id')}")
assert job_state_data.get("job_id") and manifest_data.get("run_id"), "Missing job_id or run_id!"

# Compare run directory / output directory
manifest_out_dir = manifest_data.get("config", {}).get("logging", {}).get("output_dir")
print(f"  • output_dir: job_state={job_state_data.get('run_dir')} vs manifest={manifest_out_dir}")
assert Path(job_state_data.get("run_dir")).resolve() == Path(manifest_out_dir).resolve(), "output_dir mismatch!"

# Compare status across job_state, manifest, and cancellation response
print(f"  • status: job_state={job_state_data.get('status')} vs manifest={manifest_data.get('status')} vs cancel_res={cancel_res.status.value}")
assert job_state_data.get("status") == "CANCELLED", f"Expected CANCELLED in job_state, got {job_state_data.get('status')}"
assert manifest_data.get("status") == "CANCELLED", f"Expected CANCELLED in manifest, got {manifest_data.get('status')}"
assert cancel_res.status.value == "CANCELLED", f"Expected CANCELLED in cancel_res, got {cancel_res.status.value}"

# Compare steps across job_state, manifest, and last event log
manifest_steps = manifest_data.get("final_metrics", {}).get("total_steps", job_state_data.get("current_step"))
print(f"  • current_step: job_state={job_state_data.get('current_step')}, manifest={manifest_steps}, last_event={last_event.get('step')}")
assert job_state_data.get("current_step") == last_event.get("step"), "current_step mismatch between job_state and events.jsonl!"
assert manifest_steps == last_event.get("step"), "Step mismatch between manifest and events.jsonl!"

tolerance = 1e-4
diff_loss = abs(float(job_state_data.get("latest_loss")) - float(last_event.get("loss")))
manifest_loss = float(manifest_data.get("final_metrics", {}).get("final_loss", job_state_data.get("latest_loss")))
diff_manifest_loss = abs(float(job_state_data.get("latest_loss")) - manifest_loss)
print(f"  • latest_loss: job_state={job_state_data.get('latest_loss')}, manifest={manifest_loss}, last_event={last_event.get('loss')} (diff={diff_loss:.2e}, tol={tolerance:.2e})")
assert diff_loss < tolerance, f"Loss difference {diff_loss} exceeds tolerance {tolerance}!"
assert diff_manifest_loss < tolerance, f"Manifest loss difference {diff_manifest_loss} exceeds tolerance {tolerance}!"

# Check checkpoint metrics agreement if saved
ckpt_dir = job_state_data.get("checkpoint_dir")
if ckpt_dir:
    print(f"  • checkpoint: {ckpt_dir} (exists={Path(ckpt_dir).exists()})")
    assert Path(ckpt_dir).exists(), f"Checkpoint directory {ckpt_dir} does not exist on disk!"

lifecycle_agreement = {
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
agreement_out = PROJECT_ROOT / "artifacts" / "DELTA23_LIFECYCLE_AGREEMENT.json"
agreement_out.write_text(json.dumps(lifecycle_agreement, indent=2), encoding="utf-8")
print(f"[PASS] Cross-artifact agreement verified: step={job_state_data.get('current_step')}, loss diff={diff_loss:.2e} < {tolerance:.2e}")
print(f"       Saved lifecycle agreement verification record to: {agreement_out}")
print("ALL LIFECYCLE TESTS PASSED IN INSTALLED CANDIDATE.")
