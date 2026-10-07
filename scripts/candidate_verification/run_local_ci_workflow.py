"""Local execution equivalent of .github/workflows/ci.yml with constraints.txt.
Executes complete CI pipeline locally in an isolated virtual environment:
1. Build / verify distribution wheel artifact identity (frozen candidate wheel).
2. Create dedicated isolated virtual environment (artifacts/ci_local_venv).
3. Install candidate wheel strictly adhering to constraints.txt.
4. Execute pip check to verify 0 broken dependencies.
5. Verify CLI entry point executables (train-stack --help, train-stack inspect, train-stack preflight).
6. Execute scoped smoke test suite inside the isolated environment with TRAIN_STACK_VERIFY_INSTALLED=1,
   asserting in-process site-packages provenance in pytest.
7. Record complete execution evidence to artifacts/DELTA23_CONSTRAINTS_CI_EVIDENCE.log.
"""

import os
import sys
import time
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Local CI workflow execution with explicit paths")
parser.add_argument("--project-root", type=str, default=None, help="Explicit path to project root")
parser.add_argument("--repo-root", type=str, default=None, help="Explicit path to repo root")
parser.add_argument("--wheel-path", type=str, default=None, help="Explicit path to candidate wheel")
parser.add_argument("--constraints-file", type=str, default=None, help="Explicit path to constraints.txt")
parser.add_argument("--venv-dir", type=str, default=None, help="Explicit path to CI virtual environment")
parser.add_argument("--log-path", type=str, default=None, help="Explicit path to output evidence log")
args, _ = parser.parse_known_args()

if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
elif os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

REPO_DIR = Path(args.repo_root).resolve() if args.repo_root else SCRIPT_DIR.parent.parent

if args.wheel_path:
    WHEEL_PATH = Path(args.wheel_path).resolve()
else:
    WHEEL_PATH = PROJECT_ROOT / "artifacts" / "llm_training_stack-0.1.0-py3-none-any.whl"
    if not WHEEL_PATH.exists():
        WHEEL_PATH = PROJECT_ROOT / "shared" / "llm_training_stack-0.1.0-py3-none-any.whl"

CONSTRAINTS_FILE = Path(args.constraints_file).resolve() if args.constraints_file else (REPO_DIR / "constraints.txt")
CI_VENV_DIR = Path(args.venv_dir).resolve() if args.venv_dir else (PROJECT_ROOT / "artifacts" / "ci_local_venv")
CI_VENV_PY = CI_VENV_DIR / "Scripts" / "python.exe"
CI_CLI_EXE = CI_VENV_DIR / "Scripts" / "train-stack.exe"
CI_LOG_PATH = Path(args.log_path).resolve() if args.log_path else (PROJECT_ROOT / "artifacts" / "DELTA23_CONSTRAINTS_CI_EVIDENCE.log")

UV_EXE = shutil.which("uv") or "uv"

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

def run_ci():
    log("=" * 80)
    log("CONSTRAINTS-BACKED CLEAN CI LOCAL EQUIVALENT EXECUTION (DELTA 23)")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Project Root: {PROJECT_ROOT}")
    log(f"Repository Root: {REPO_DIR}")
    log(f"Constraints File: {CONSTRAINTS_FILE}")
    log(f"Candidate Wheel: {WHEEL_PATH}")
    log(f"Evidence Log: {CI_LOG_PATH}")
    log("=" * 80)

    # 1. Verify Wheel Artifact Identity
    log("\n--- STEP 1: VERIFY DISTRIBUTION WHEEL ARTIFACT ---")
    assert WHEEL_PATH.exists(), f"Wheel artifact not found at {WHEEL_PATH}"
    assert CONSTRAINTS_FILE.exists(), f"Constraints file not found at {CONSTRAINTS_FILE}"

    wheel_bytes = WHEEL_PATH.read_bytes()
    wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()
    log(f"Wheel Size: {len(wheel_bytes)} bytes")
    log(f"Wheel SHA256: {wheel_sha}")
    assert len(wheel_bytes) == 83637, f"Unexpected wheel size: {len(wheel_bytes)}"
    assert wheel_sha == "dbaabab553387ccad26f8f724446de3c6746ed8039b11a30eb8d7cce1d0865a1"
    log("[PASS] Wheel artifact verified: 83,637 bytes, SHA256 dbaabab5...")

    # 2. Create Isolated Virtual Environment
    log("\n--- STEP 2: CREATE ISOLATED CI VIRTUAL ENVIRONMENT ---")
    if CI_VENV_DIR.exists():
        log(f"Removing existing CI virtualenv at {CI_VENV_DIR} for clean execution...")
        for attempt in range(5):
            try:
                shutil.rmtree(CI_VENV_DIR)
                break
            except Exception:
                time.sleep(1)
        if CI_VENV_DIR.exists():
            shutil.rmtree(CI_VENV_DIR, ignore_errors=True)
            time.sleep(1)

    rel_venv = os.path.relpath(CI_VENV_DIR, PROJECT_ROOT)
    run_cmd(f'"{UV_EXE}" venv "{rel_venv}" --python 3.11')
    assert CI_VENV_PY.exists(), f"Failed to create virtual environment Python at {CI_VENV_PY}"
    log(f"[PASS] Isolated virtualenv created: {CI_VENV_DIR}")

    # 3. Install Candidate Wheel with Constraints
    log("\n--- STEP 3: INSTALL CANDIDATE WHEEL WITH CONSTRAINTS.TXT ---")
    rel_py = os.path.relpath(CI_VENV_PY, PROJECT_ROOT)
    rel_wheel = os.path.relpath(WHEEL_PATH, PROJECT_ROOT)
    rel_constraints = os.path.relpath(CONSTRAINTS_FILE, PROJECT_ROOT)

    run_cmd(
        f'"{UV_EXE}" pip install --python "{rel_py}" "{rel_wheel}" pip -c "{rel_constraints}"'
    )
    assert CI_CLI_EXE.exists(), f"CLI executable train-stack.exe not found at {CI_CLI_EXE}"
    log("[PASS] Candidate wheel and constrained dependencies successfully installed.")

    # 4. pip check
    log("\n--- STEP 4: PIP CHECK DEPENDENCY CONSISTENCY ---")
    run_cmd(f'"{UV_EXE}" pip check --python "{rel_py}"')
    run_cmd(f'"{rel_py}" -m pip check')
    log("[PASS] pip check passed: 0 broken requirements.")

    # 5. CLI Entry Point Verification
    log("\n--- STEP 5: CLI ENTRY POINT VERIFICATION ---")
    rel_cli = os.path.relpath(CI_CLI_EXE, PROJECT_ROOT)
    run_cmd(f'"{rel_cli}" --help')
    run_cmd(f'"{rel_cli}" inspect')
    run_cmd(f'"{rel_cli}" preflight --config "shared/llm-training-stack/examples/training_config.yaml"')
    log("[PASS] CLI entry points verified.")

    # 6. Install test runner with constraints and execute smoke test suite
    log("\n--- STEP 6: SCOPED SMOKE TEST SUITE EXECUTION WITH INSTALLED PROVENANCE ---")
    run_cmd(f'"{CI_VENV_PY}" -m pip install pytest pytest-cov -c "{CONSTRAINTS_FILE}"')

    # Create an isolated test working directory to guarantee repository cwd does not leak into sys.path
    isolated_test_dir = PROJECT_ROOT / "artifacts" / "ci_test_workspace"
    isolated_test_dir.mkdir(parents=True, exist_ok=True)

    test_env = os.environ.copy()
    test_env["TRAIN_STACK_VERIFY_INSTALLED"] = "1"
    # Ensure PYTHONPATH does not point to repo root
    test_env.pop("PYTHONPATH", None)

    # Copy test files into isolated workspace
    test_src_dir = REPO_DIR / "tests"
    test_dest_dir = isolated_test_dir / "tests"
    if test_dest_dir.exists():
        shutil.rmtree(test_dest_dir, ignore_errors=True)
    shutil.copytree(test_src_dir, test_dest_dir)

    test_targets = [
        "tests/test_config.py",
        "tests/test_provenance.py",
        "tests/test_cli.py",
    ]

    log(f"Running pytest from isolated workspace: {isolated_test_dir}")
    log(f"Verifying in-process site-packages provenance assertion (test_installed_package_provenance_site_packages)...")

    run_cmd(
        f'"{CI_VENV_PY}" -m pytest {" ".join(test_targets)} -v',
        cwd=str(isolated_test_dir),
        env=test_env,
    )
    log("[PASS] In-process pytest test suite passed with site-packages provenance verified!")

    log("\n" + "=" * 80)
    log("CONSTRAINTS-BACKED LOCAL CI WORKFLOW PASSED (100% SUCCESS)")
    log("=" * 80)

    # Save log
    CI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CI_LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nDurable CI log written to: {CI_LOG_PATH} ({CI_LOG_PATH.stat().st_size} bytes)")

    # Keep CONSTRAINTS_CI_EVIDENCE.log synchronized
    legacy_log = PROJECT_ROOT / "artifacts" / "CONSTRAINTS_CI_EVIDENCE.log"
    legacy_log.write_text("\n".join(log_lines), encoding="utf-8")

    # Sync to shared
    shared_ci_log = PROJECT_ROOT / "shared" / "DELTA23_CONSTRAINTS_CI_EVIDENCE.log"
    shared_ci_log.write_text(CI_LOG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"Synchronized CI evidence to: {shared_ci_log}")

if __name__ == "__main__":
    run_ci()
