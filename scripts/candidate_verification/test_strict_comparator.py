"""Strict RunComparator compatibility and negative rejection verification.
Verifies:
1. Retained compatible runs pass strict comparison without error.
2. Incompatible runs fail strict comparison with explicit ValueError across all 5 rejection conditions:
   a. Mismatched held-out dataset fingerprints
   b. Mismatched tokenizer identities
   c. Mismatched masking schemes
   d. Mismatched max_seq_length
   e. Missing evaluation reports
3. Non-strict comparison generates compatibility warning banner without raising.
No adaptation re-runs required.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

import os
import argparse

# Project root resolution: explicit CLI argument -> env var -> auto-detection
parser = argparse.ArgumentParser(description="Strict RunComparator compatibility verification")
parser.add_argument("--project-root", type=str, default=None, help="Explicit path to project root")
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).resolve().parent
if args.project_root:
    PROJECT_ROOT = Path(args.project_root).resolve()
elif os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    PROJECT_ROOT = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "artifacts").is_dir() and (p / "shared").is_dir()), SCRIPT_DIR.parents[3]).resolve()

from llm_training_stack.eval.comparator import RunComparator

def run_test():
    print("=" * 80)
    print("STRICT RUN COMPARATOR & INCOMPATIBLE NEGATIVE VERIFICATION")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    # 1. Verify retained candidate baseline and adapted run with strict_compatibility=True
    cand_base = PROJECT_ROOT / "artifacts" / "installed_candidate_baseline"
    cand_run = PROJECT_ROOT / "artifacts" / "installed_candidate_run"

    print("\n[Test 1/6] Strict comparison on retained installed candidate runs...")
    assert cand_base.exists(), f"Missing {cand_base}"
    assert cand_run.exists(), f"Missing {cand_run}"

    res = RunComparator.compare_runs([cand_base, cand_run], strict_compatibility=True)
    assert res["compatibility"]["is_comparable"] is True
    assert len(res["compatibility"]["reasons"]) == 0
    print(f"[PASS] Retained candidate comparison is strictly compatible (2 runs evaluated).")

    # Also verify retained real smoke runs if present
    smoke_base = PROJECT_ROOT / "artifacts" / "real_smoke_baseline"
    smoke_run = PROJECT_ROOT / "artifacts" / "real_smoke_run"
    if smoke_base.exists() and smoke_run.exists():
        res_smoke = RunComparator.compare_runs([smoke_base, smoke_run], strict_compatibility=True)
        assert res_smoke["compatibility"]["is_comparable"] is True
        print(f"[PASS] Retained real smoke run comparison is strictly compatible.")

    # Create temporary fixture runs for isolated negative testing
    tmp_dir = Path(tempfile.mkdtemp(prefix="comparator_negative_test_"))
    try:
        # Load baseline files as templates
        base_manifest = json.loads((cand_base / "manifest.json").read_text(encoding="utf-8"))
        base_eval = json.loads((cand_base / "eval_report.json").read_text(encoding="utf-8"))

        def make_fixture(name, eval_overrides=None, manifest_overrides=None, include_eval=True):
            f_dir = tmp_dir / name
            f_dir.mkdir(parents=True, exist_ok=True)
            m = dict(base_manifest)
            m["run_id"] = name
            if manifest_overrides:
                m.update(manifest_overrides)
            (f_dir / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            if include_eval:
                ev = dict(base_eval)
                if eval_overrides:
                    ev.update(eval_overrides)
                (f_dir / "eval_report.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
            return f_dir

        base_fix = make_fixture("valid_base")

        # 2. Negative: Mismatched dataset fingerprints
        print("\n[Test 2/6] Negative: Mismatched dataset fingerprints...")
        diff_dataset_fix = make_fixture("diff_dataset", eval_overrides={"dataset_fingerprint": "mismatched_deadbeef9999"})
        try:
            RunComparator.compare_runs([base_fix, diff_dataset_fix], strict_compatibility=True)
            assert False, "Strict comparison must fail on mismatched dataset fingerprints!"
        except ValueError as e:
            assert "Held-out evaluation datasets differ across runs" in str(e)
            print(f"[PASS] Caught expected ValueError: {e}")

        # 3. Negative: Mismatched tokenizer identities
        print("\n[Test 3/6] Negative: Mismatched tokenizer identities...")
        diff_tok_fix = make_fixture("diff_tok", eval_overrides={"tokenizer_name_or_path": "mismatched-gpt2-tokenizer"})
        try:
            RunComparator.compare_runs([base_fix, diff_tok_fix], strict_compatibility=True)
            assert False, "Strict comparison must fail on mismatched tokenizer identities!"
        except ValueError as e:
            assert "Tokenizers differ across runs" in str(e)
            print(f"[PASS] Caught expected ValueError: {e}")

        # 4. Negative: Mismatched masking schemes
        print("\n[Test 4/6] Negative: Mismatched masking schemes...")
        diff_mask_fix = make_fixture("diff_mask", eval_overrides={"masking_scheme": "all_tokens"})
        try:
            RunComparator.compare_runs([base_fix, diff_mask_fix], strict_compatibility=True)
            assert False, "Strict comparison must fail on mismatched masking schemes!"
        except ValueError as e:
            assert "Masking schemes differ across runs" in str(e)
            print(f"[PASS] Caught expected ValueError: {e}")

        # 5. Negative: Mismatched max_seq_length
        print("\n[Test 5/6] Negative: Mismatched max_seq_length...")
        diff_len_fix = make_fixture("diff_len", eval_overrides={"max_seq_length": 512})
        try:
            RunComparator.compare_runs([base_fix, diff_len_fix], strict_compatibility=True)
            assert False, "Strict comparison must fail on mismatched max_seq_length!"
        except ValueError as e:
            assert "Evaluation max_seq_length differs across runs" in str(e)
            print(f"[PASS] Caught expected ValueError: {e}")

        # 6. Negative: Missing evaluation report & non-strict warning banner check
        print("\n[Test 6/6] Negative: Missing eval report & non-strict fallback check...")
        missing_eval_fix = make_fixture("missing_eval", include_eval=False)
        try:
            RunComparator.compare_runs([base_fix, missing_eval_fix], strict_compatibility=True)
            assert False, "Strict comparison must fail on missing evaluation report!"
        except ValueError as e:
            assert "Candidate runs missing held-out evaluation report" in str(e)
            print(f"[PASS] Caught expected ValueError for missing eval: {e}")

        # Verify non-strict execution handles incompatible runs by embedding warning banner
        non_strict_res = RunComparator.compare_runs([base_fix, diff_dataset_fix], strict_compatibility=False)
        assert non_strict_res["compatibility"]["is_comparable"] is False
        assert "Compatibility Notice" in non_strict_res["markdown_report"]
        print("[PASS] Non-strict comparison correctly emitted Compatibility Notice warning block.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("ALL STRICT RUN COMPARATOR CHECKS PASSED (100% SUCCESS)")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
