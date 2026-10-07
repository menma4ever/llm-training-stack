"""Audits candidate installation provenance, site-packages isolation, and framework versions.
Project-relative, portable, and repeatable.
"""

import sys
import os
import hashlib
from pathlib import Path

import argparse

# Project root resolution: explicit CLI argument -> env var -> auto-detection
parser = argparse.ArgumentParser(description="Candidate provenance and isolation audit")
parser.add_argument("--project-root", type=str, default=None, help="Explicit path to project root")
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).resolve().parent
if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
elif os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

print("=" * 80)
print("CANDIDATE PROVENANCE & SITE-PACKAGES ISOLATION AUDIT")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("=" * 80)

# 1. Audit llm_training_stack origin
import llm_training_stack

pkg_file = Path(llm_training_stack.__file__).resolve()
print(f"MODULE_FILE: {pkg_file}")
print(f"SYS_PATH_0: {sys.path[0]}")

assert "site-packages" in str(pkg_file), f"llm_training_stack not imported from site-packages! ({pkg_file})"
assert "shared\\llm-training-stack" not in str(pkg_file) and "shared/llm-training-stack" not in str(pkg_file), (
    f"llm_training_stack was loaded from source repository rather than installed wheel! ({pkg_file})"
)
print("[PASS] llm_training_stack provenance verified: loaded exclusively from site-packages.")

# 2. Audit Core Framework Versions
import torch
import transformers
import accelerate
import peft
import trl
import pydantic
import safetensors
import typer

print(f"TORCH_VERSION: {torch.__version__}")
print(f"TRANSFORMERS_VERSION: {transformers.__version__}")
print(f"ACCELERATE_VERSION: {accelerate.__version__}")
print(f"PEFT_VERSION: {peft.__version__}")
print(f"TRL_VERSION: {trl.__version__}")
print(f"PYDANTIC_VERSION: {pydantic.__version__}")
print(f"SAFETENSORS_VERSION: {safetensors.__version__}")
print(f"TYPER_VERSION: {typer.__version__}")

# 3. Candidate Wheel Verification
wheel_path = PROJECT_ROOT / "artifacts" / "llm_training_stack-0.1.0-py3-none-any.whl"
if not wheel_path.exists():
    wheel_path = PROJECT_ROOT / "shared" / "llm_training_stack-0.1.0-py3-none-any.whl"

assert wheel_path.exists(), f"Candidate wheel not found at {wheel_path}"
wheel_bytes = wheel_path.read_bytes()
wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()
print(f"CANDIDATE_WHEEL_PATH: {wheel_path}")
print(f"CANDIDATE_WHEEL_SIZE: {len(wheel_bytes)} bytes")
print(f"CANDIDATE_WHEEL_SHA256: {wheel_sha}")

assert len(wheel_bytes) == 83637, f"Unexpected candidate wheel size: {len(wheel_bytes)} (expected 83637)"
assert wheel_sha == "dbaabab553387ccad26f8f724446de3c6746ed8039b11a30eb8d7cce1d0865a1", (
    f"Unexpected candidate wheel SHA256: {wheel_sha}"
)
print("[PASS] Candidate wheel identity verified: 83,637 bytes, SHA256: dbaabab5...")
print("=" * 80)
print("PROVENANCE AUDIT PASSED (100% SUCCESS)")
print("=" * 80)
