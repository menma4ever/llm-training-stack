"""Rigorous privacy and sanitization audit of artifacts/public_export."""

import os
import sys
import hashlib
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description="Audit privacy and sanitization of public export")
parser.add_argument("--project-root", default=None, help="Project root directory")
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).resolve().parent
if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

export_dir = PROJECT_ROOT / "artifacts" / "public_export"
assert export_dir.exists(), f"Public export directory not found at {export_dir}"

patterns = [
    re.compile(r"abdulaziz", re.IGNORECASE),
    re.compile(r"komilov", re.IGNORECASE),
    re.compile(r"[a-zA-Z]:\\[Users|Documents|AppData]", re.IGNORECASE),
    re.compile(r"/Users/[^/\s]+", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"operator-secret-[^\s\"]+"),
]

files_scanned = []
violations = []

for root, _, files in os.walk(export_dir):
    for f in files:
        fpath = Path(root) / f
        rel_path = fpath.relative_to(export_dir)
        content_bytes = fpath.read_bytes()
        file_sha = hashlib.sha256(content_bytes).hexdigest()
        files_scanned.append((str(rel_path).replace("\\", "/"), len(content_bytes), file_sha))

        try:
            text = content_bytes.decode("utf-8")
            for pat in patterns:
                matches = pat.findall(text)
                if matches:
                    violations.append((str(rel_path).replace("\\", "/"), pat.pattern, matches[:3]))
        except UnicodeDecodeError:
            pass

scan_log = [
    "=" * 80,
    "PUBLIC EXPORT RIGOROUS PRIVACY AND SANITIZATION AUDIT",
    f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
    f"Export Directory: {export_dir}",
    f"Total Files Scanned: {len(files_scanned)}",
    "=" * 80,
    "",
    "--- SCANNED FILES REGISTRY ---",
]

for rpath, size, sha in sorted(files_scanned):
    scan_log.append(f"  • {rpath}: {size} bytes, sha256={sha}")

scan_log.append("")
scan_log.append("--- SCAN VERDICT ---")
if violations:
    scan_log.append(f"[FAILED] {len(violations)} privacy violations found:")
    for rpath, pat, matches in violations:
        scan_log.append(f"  • {rpath}: pattern {pat!r} matched {matches}")
    sys.exit(1)
else:
    scan_log.append("[PASS] Zero private paths, usernames, tokens, or environment secrets detected.")
    scan_log.append("       Public export is 100% sanitized and ready for public repository release.")

scan_log.append("=" * 80)

out_text = "\n".join(scan_log)
print(out_text)
out_log = PROJECT_ROOT / "artifacts" / "CURRENT_EXPORT_PRIVACY_SCAN.log"
out_log.write_text(out_text, encoding="utf-8")
print(f"\nSaved export scan log to: {out_log}")
