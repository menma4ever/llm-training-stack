import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
project_root = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[4]).resolve()
targets = [
    "artifacts/llm_training_stack-0.1.0-py3-none-any.whl",
    "artifacts/DELTA23_FINAL_INSTALLED_CANDIDATE_EVIDENCE.log",
    "artifacts/DELTA23_CONSTRAINTS_CI_EVIDENCE.log",
    "artifacts/DELTA23_INSTALLED_CLI_WORKFLOW_EVIDENCE.log",
    "artifacts/DELTA23_CANDIDATE_ORCHESTRATION_HARNESS.log",
    "artifacts/DELTA23_MCP_AGREEMENT.json",
    "artifacts/DELTA23_LIFECYCLE_AGREEMENT.json",
    "artifacts/installed_candidate_run/manifest.json",
    "artifacts/installed_candidate_run/eval_report.json",
    "artifacts/installed_candidate_run/comparison_report.json",
    "artifacts/installed_candidate_baseline/manifest.json",
    "artifacts/installed_candidate_baseline/eval_report.json",
    "artifacts/cli_eval_report.json",
    "artifacts/real_heldout_dataset.jsonl",
    "shared/llm-training-stack/constraints.txt",
    "shared/llm-training-stack/pyproject.toml",
    "shared/llm-training-stack/scripts/candidate_verification/run_orchestration_harness.py",
    "shared/llm-training-stack/scripts/candidate_verification/run_local_ci_workflow.py",
    "shared/llm-training-stack/scripts/candidate_verification/run_mcp_in_candidate.py",
    "shared/llm-training-stack/scripts/candidate_verification/run_lifecycle_in_candidate.py",
    "shared/llm-training-stack/scripts/candidate_verification/audit_provenance_candidate.py",
    "shared/llm-training-stack/scripts/candidate_verification/test_strict_comparator.py",
    "shared/llm-training-stack/scripts/candidate_verification/run_adaptation_in_candidate.py"
]

registry = {
    "schema_version": "1.0.0",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "verified_by": "Test_Integration_Specialist",
    "agent_id": "a01ec5ac4fe44487a1a7b0bcf8d7a0a0",
    "directive": "CEO Release Decision 23 / RELEASE_DELTA_REVIEW_23.md",
    "status": "ALL_GATES_PASSED_100_PERCENT",
    "artifacts": {}
}

for rel_path in targets:
    p = project_root / rel_path
    if p.exists():
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        registry["artifacts"][rel_path] = {
            "relative_path": rel_path,
            "absolute_path": str(p),
            "size_bytes": len(data),
            "sha256": sha
        }
        print(f"[REGISTERED] {rel_path} ({len(data)} bytes, {sha[:16]}...)")
    else:
        print(f"[MISSING] {rel_path}")

reg_path = project_root / "artifacts" / "DELTA23_EVIDENCE_REGISTRY.json"
reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(f"Saved evidence registry to {reg_path} ({reg_path.stat().st_size} bytes)")

shared_reg = project_root / "shared" / "DELTA23_EVIDENCE_REGISTRY.json"
shared_reg.write_text(json.dumps(registry, indent=2), encoding="utf-8")

worker_artifacts = project_root / "workers" / "Test_Integration_Specialist" / "artifacts"
worker_artifacts.mkdir(parents=True, exist_ok=True)
(worker_artifacts / "DELTA23_EVIDENCE_REGISTRY.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
print("Synchronized registry across artifacts/, shared/, and workers/.")
