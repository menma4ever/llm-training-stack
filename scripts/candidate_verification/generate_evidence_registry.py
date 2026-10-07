"""Generate immutable DELTA23 Evidence Registry JSON and Markdown."""

import os
import sys
import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate immutable DELTA23 evidence registry")
parser.add_argument("--project-root", default=None, help="Project root directory")
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).resolve().parent
if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

artifacts_to_register = [
    # Candidate Package
    ("artifacts/llm_training_stack-0.1.0-py3-none-any.whl", "Candidate wheel package with synchronized CLI distinction, site-packages provenance test, and subprocess stdin fix"),

    # Master Candidate Verification Logs
    ("artifacts/DELTA23_FINAL_INSTALLED_CANDIDATE_EVIDENCE.log", "Master candidate verification log (8 gates passed, provenance, adaptation, lifecycle, comparator, MCP STDIO, active CLI cancel)"),
    ("artifacts/DELTA23_CANDIDATE_ORCHESTRATION_HARNESS.log", "Candidate orchestration harness execution log"),
    ("artifacts/DELTA23_INSTALLED_CLI_WORKFLOW_EVIDENCE.log", "Installed CLI workflow evidence log with active in-progress job cancellation and terminal cancellation"),
    ("artifacts/DELTA23_CONSTRAINTS_CI_EVIDENCE.log", "Constrained local CI workflow log with constraints.txt resolution, pip check (0 broken), and 18 pytest tests"),

    # Cross-Artifact Accord Records
    ("artifacts/DELTA23_LIFECYCLE_AGREEMENT.json", "Cross-artifact lifecycle agreement record verifying status=CANCELLED, strict PID exit, step/loss accord within 1e-4 tolerance"),
    ("artifacts/DELTA23_MCP_AGREEMENT.json", "Cross-artifact FastMCP agreement record verifying status=CANCELLED, strict PID exit, step/loss accord within 1e-4 tolerance"),
    ("artifacts/mcp_probe_measured_result.json", "Empirical MCP probe measured schema output with truthful non-representative CPU warning and peak RSS"),

    # Documentation & Specifications
    ("artifacts/COMPATIBILITY.md", "Hardware & environment compatibility matrix reconciled to constraints.txt inventory and explicit multi-OS CI scope"),
    ("shared/llm-training-stack/constraints.txt", "Strict dependency constraints file pinning exact tested versions"),

    # Retained Historical Proofs
    ("artifacts/NATIVE_UPSTREAM_RESUME_EVIDENCE.log", "Retained native upstream resume continuity evidence log"),
    ("artifacts/STRICT_COMPARATOR_EVIDENCE.log", "Retained strict comparator evidence log with five negative cases and report checks"),
    ("artifacts/CURRENT_EXPORT_PRIVACY_SCAN.log", "Rigorous privacy and secret scan log verifying 0 leaks across public export"),

    # Public Export Artifacts
    ("artifacts/public_export/data/real_heldout_dataset.jsonl", "Sanitized public held-out evaluation dataset"),
    ("artifacts/public_export/data/real_sft_dataset.jsonl", "Sanitized public SFT training dataset"),
    ("artifacts/public_export/installed_candidate_run/eval_report.json", "Sanitized public candidate evaluation report"),
    ("artifacts/public_export/installed_candidate_run/manifest.json", "Sanitized public candidate training manifest"),
    ("artifacts/public_export/installed_candidate_run/comparison_report.json", "Sanitized public comparison report"),
    ("artifacts/public_export/installed_candidate_run/events.jsonl", "Sanitized public candidate event log"),
    ("artifacts/public_export/installed_candidate_run/launch_manifest.json", "Sanitized public candidate launch manifest"),
    ("artifacts/public_export/installed_candidate_run/baseline_eval_report.json", "Sanitized public baseline evaluation report"),
    ("artifacts/public_export/real_smoke_run/manifest.json", "Sanitized public smoke run manifest"),
    ("artifacts/public_export/real_smoke_run/events.jsonl", "Sanitized public smoke run event log"),
    ("artifacts/public_export/ARTIFACT_FINGERPRINTS.json", "Public export artifact fingerprints registry"),
]

registry = {
    "registry_version": "1.0.0",
    "title": "DELTA 23 Final Immutable Evidence Registry",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "project_root": str(PROJECT_ROOT),
    "artifacts": []
}

for rel_path, desc in artifacts_to_register:
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        print(f"MISSING: {rel_path}")
        sys.exit(1)
    b = p.read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    registry["artifacts"].append({
        "path": rel_path.replace("\\", "/"),
        "description": desc,
        "bytes": len(b),
        "sha256": sha,
    })

# Write JSON
json_out = PROJECT_ROOT / "artifacts" / "DELTA23_EVIDENCE_REGISTRY.json"
json_out.write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(f"Wrote JSON registry: {json_out} ({len(registry['artifacts'])} items)")

# Write Markdown
md_lines = [
    "# DELTA 23 Final Immutable Evidence Registry",
    "",
    f"**Generated (UTC)**: `{registry['timestamp_utc']}`  ",
    "**Scope**: Exact physical byte count and SHA256 cryptographic digest of all DELTA 23 remediation evidence artifacts.  ",
    "",
    "| Artifact Path | Size (Bytes) | SHA256 Cryptographic Hash | Description |",
    "|---|---|---|---|",
]

for item in registry["artifacts"]:
    p = item["path"]
    sz = f"{item['bytes']:,}"
    full_sha = item["sha256"]
    desc = item["description"]
    md_lines.append(f"| `{p}` | {sz} | `{full_sha}` | {desc} |")

md_lines.append("")
md_lines.append("---")
md_lines.append("### Raw-to-Sanitized Path Mapping in Public Export")
md_lines.append("")
md_lines.append("| Raw Candidate File | Public Export Path | Sanitization Transformation |")
md_lines.append("|---|---|---|")
md_lines.append("| `artifacts/installed_candidate_run/manifest.json` | `artifacts/public_export/installed_candidate_run/manifest.json` | Replaced absolute workspace paths with relative `./runs/...` and `./data/...` paths |")
md_lines.append("| `artifacts/installed_candidate_run/eval_report.json` | `artifacts/public_export/installed_candidate_run/eval_report.json` | Replaced local model path with relative checkpoint reference |")
md_lines.append("| `artifacts/real_sft_dataset.jsonl` | `artifacts/public_export/data/real_sft_dataset.jsonl` | Copied clean raw jsonl training samples |")
md_lines.append("| `artifacts/real_heldout_dataset.jsonl` | `artifacts/public_export/data/real_heldout_dataset.jsonl` | Copied clean raw jsonl held-out samples |")
md_lines.append("")

md_out = PROJECT_ROOT / "artifacts" / "DELTA23_EVIDENCE_REGISTRY.md"
md_out.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Wrote Markdown registry: {md_out}")
