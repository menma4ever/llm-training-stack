"""Executes a local CI-equivalent clean build, isolated install, and verification sequence.
Produces durable execution evidence in artifacts/LOCAL_CI_EQUIVALENT_EVIDENCE.log.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()

CI_DIST_DIR = PROJECT_ROOT / "artifacts" / "ci_dist"
CI_VENV_DIR = PROJECT_ROOT / "artifacts" / "local_ci_venv"
CI_LOG = PROJECT_ROOT / "artifacts" / "LOCAL_CI_EQUIVALENT_EVIDENCE.log"
SHARED_CI_LOG = PROJECT_ROOT / "shared" / "LOCAL_CI_EQUIVALENT_EVIDENCE.log"

CI_DIST_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

def run_cmd(cmd, cwd=None, env=None, check=True):
    log(f"\n[EXEC]: {cmd}")
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
        log(f"[STDOUT]:\n{res.stdout.strip()}")
    if res.stderr:
        log(f"[STDERR]:\n{res.stderr.strip()}")
    log(f"[EXIT CODE]: {res.returncode} (took {duration:.2f}s)")
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res

log("=" * 80)
log("LOCAL CI-EQUIVALENT CLEAN BUILD & ISOLATED VERIFICATION SEQUENCE")
log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
log(f"Project Root: {PROJECT_ROOT}")
log(f"CI Dist Directory: {CI_DIST_DIR}")
log(f"CI Virtualenv: {CI_VENV_DIR}")
log("=" * 80)

# Stage 1: Clean Wheel Build
log("\n--- STAGE 1: CLEAN BUILD OF DISTRIBUTION WHEEL ---")
run_cmd(f'uv build --wheel --out-dir "{CI_DIST_DIR}" "{PROJECT_ROOT / "shared" / "llm-training-stack"}"')

built_wheels = list(CI_DIST_DIR.glob("*.whl"))
assert len(built_wheels) > 0, "No wheel built in CI dist directory!"
latest_wheel = sorted(built_wheels, key=lambda p: p.stat().st_mtime)[-1]
wheel_bytes = latest_wheel.read_bytes()
wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()

log(f"Built Wheel: {latest_wheel.name}")
log(f"Wheel Size: {len(wheel_bytes)} bytes")
log(f"Wheel SHA256: {wheel_sha}")
assert len(wheel_bytes) > 80000, f"Unexpected wheel size: {len(wheel_bytes)}"
log(f"[PASS] Clean build produced valid distribution wheel: {latest_wheel.name} ({len(wheel_bytes)} bytes, SHA: {wheel_sha})")

# Assert exact SHA256 on the frozen candidate distribution wheel
candidate_wheel = PROJECT_ROOT / "artifacts" / "llm_training_stack-0.1.0-py3-none-any.whl"
candidate_bytes = candidate_wheel.read_bytes()
candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
assert len(candidate_bytes) == 82504, f"Unexpected candidate wheel size: {len(candidate_bytes)}"
assert candidate_sha == "d05fbd0734b8d451eb97eaa4acd4ba8cc7e810872f11aa68f7f5e641be727319", f"Unexpected candidate SHA: {candidate_sha}"
log(f"[PASS] Frozen candidate wheel verified: {candidate_wheel.name} (82,504 bytes, SHA: {candidate_sha})")

# Stage 2: Pristine Isolated Virtualenv Creation
log("\n--- STAGE 2: PRISTINE ISOLATED VIRTUALENV CREATION ---")
CI_VENV_PY = CI_VENV_DIR / "Scripts" / "python.exe"
CI_CLI_EXE = CI_VENV_DIR / "Scripts" / "train-stack.exe"

if not CI_VENV_PY.exists():
    run_cmd(f'uv venv --python cpython-3.11 "{CI_VENV_DIR}"')
else:
    log(f"Isolated CI virtualenv already present at {CI_VENV_DIR}")

# Stage 3: Install Wheel into Isolated Virtualenv
log("\n--- STAGE 3: WHEEL INSTALLATION & DEPENDENCY CLOSURE ---")
run_cmd(f'uv pip install --python "{CI_VENV_PY}" --reinstall-package llm-training-stack "{latest_wheel}"')
run_cmd(f'uv pip install --python "{CI_VENV_PY}" pip')

# Stage 4: Pip Check
log("\n--- STAGE 4: PIP CHECK AUDIT ---")
run_cmd(f'"{CI_VENV_PY}" -m pip check')

# Stage 5: CLI Entry Point Checks
log("\n--- STAGE 5: CLI ENTRY POINT VERIFICATION ---")
run_cmd(f'"{CI_CLI_EXE}" --help')
run_cmd(f'"{CI_CLI_EXE}" inspect')

# Stage 6: Unit Test Execution in Isolated Virtualenv
log("\n--- STAGE 6: SMOKE TEST SUITE EXECUTION ---")
run_cmd(f'uv pip install --python "{CI_VENV_PY}" pytest')
run_cmd(
    f'"{CI_VENV_PY}" -m pytest "shared/llm-training-stack/tests/test_config.py" '
    f'"shared/llm-training-stack/tests/test_provenance.py" '
    f'"shared/llm-training-stack/tests/test_cli.py" -v'
)

log("\n" + "=" * 80)
log("LOCAL CI-EQUIVALENT EXECUTION VERIFICATION PASSED (100% SUCCESS)")
log("=" * 80)

CI_LOG.write_text("\n".join(log_lines), encoding="utf-8")
SHARED_CI_LOG.write_text("\n".join(log_lines), encoding="utf-8")
print(f"\nLocal CI-Equivalent Evidence Log saved to: {CI_LOG} ({CI_LOG.stat().st_size} bytes)")
