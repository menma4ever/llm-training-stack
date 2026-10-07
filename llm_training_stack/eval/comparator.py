"""Multi-run comparison and reproducible benchmark reporting with strict compatibility validation."""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


class RunComparator:
    """Compares multiple training runs by ingesting manifests, evaluation reports, and structured events."""

    @staticmethod
    def load_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
        path = Path(run_dir)
        manifest_file = path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found in {path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Inspect events.jsonl for loss trajectory
        events_file = path / "events.jsonl"
        steps_losses = []
        if events_file.exists():
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            if "loss" in event:
                                steps_losses.append((event.get("step", 0), event["loss"]))
                        except Exception:
                            pass

        # Inspect held-out evaluation report if present
        eval_report_file = path / "eval_report.json"
        eval_report = None
        if eval_report_file.exists():
            try:
                with open(eval_report_file, "r", encoding="utf-8") as f:
                    eval_report = json.load(f)
            except Exception:
                pass

        return {
            "path": str(path),
            "manifest": manifest,
            "eval_report": eval_report,
            "loss_trajectory": steps_losses,
        }

    @classmethod
    def check_compatibility(cls, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates comparability across candidate benchmark runs.
        Ensures all candidate runs possess evaluation reports with identical dataset fingerprints,
        matching tokenizer identities, and consistent masking schemes.
        Cannot claim comparability from absent metadata.
        """
        if len(runs_data) <= 1:
            return {"is_comparable": True, "reasons": []}

        reasons = []

        # 1. Require evaluation report presence for all candidate runs
        missing_reports = [r.get("path", "unknown") for r in runs_data if not r.get("eval_report")]
        if missing_reports:
            reasons.append(
                f"Candidate runs missing held-out evaluation report (eval_report.json): {missing_reports}. "
                "Cannot claim comparability from absent evaluation metadata."
            )
            return {"is_comparable": False, "reasons": reasons}

        # 2. Check dataset fingerprints in evaluation reports
        eval_fingerprints = {}
        for r in runs_data:
            report = r.get("eval_report", {})
            fp = report.get("dataset_fingerprint")
            if not fp:
                reasons.append(f"Run {r.get('path')} is missing dataset_fingerprint in eval_report.json.")
            else:
                eval_fingerprints[r.get("path")] = fp

        unique_fps = set(eval_fingerprints.values())
        if len(unique_fps) > 1:
            reasons.append(
                f"Held-out evaluation datasets differ across runs (fingerprints: {list(unique_fps)}). "
                "Perplexity and eval loss cannot be compared directly."
            )

        # 3. Check tokenizers
        tokenizers = {}
        for r in runs_data:
            report = r.get("eval_report", {})
            tok = report.get("tokenizer_name_or_path")
            if not tok:
                m = r.get("manifest", {})
                tok = m.get("config", {}).get("model", {}).get("tokenizer_name_or_path")
            if not tok:
                reasons.append(f"Run {r.get('path')} is missing tokenizer identity in eval_report/manifest.")
            else:
                tokenizers[r.get("path")] = tok

        unique_toks = set(tokenizers.values())
        if len(unique_toks) > 1:
            reasons.append(
                f"Tokenizers differ across runs ({list(unique_toks)}). "
                "Per-token cross-entropy and perplexity values are not comparable."
            )

        # 4. Check masking schemes
        masking_schemes = {}
        for r in runs_data:
            report = r.get("eval_report", {})
            scheme = report.get("masking_scheme")
            if not scheme:
                reasons.append(f"Run {r.get('path')} is missing masking_scheme in eval_report.json.")
            else:
                masking_schemes[r.get("path")] = scheme

        unique_schemes = set(masking_schemes.values())
        if len(unique_schemes) > 1:
            reasons.append(
                f"Masking schemes differ across runs ({list(unique_schemes)}). "
                "Evaluations using response_only cannot be compared with all_tokens."
            )

        # 5. Check sequence length protocol (max_seq_length)
        seq_lengths = {}
        for r in runs_data:
            report = r.get("eval_report", {})
            seq_len = report.get("max_seq_length")
            if seq_len is None:
                reasons.append(f"Run {r.get('path')} is missing max_seq_length in eval_report.json.")
            else:
                seq_lengths[r.get("path")] = seq_len

        unique_lengths = set(seq_lengths.values())
        if len(unique_lengths) > 1:
            reasons.append(
                f"Evaluation max_seq_length differs across runs ({list(unique_lengths)}). "
                "Per-token evaluation metrics with differing sequence truncation cannot be compared."
            )

        return {
            "is_comparable": len(reasons) == 0,
            "reasons": reasons,
        }

    @classmethod
    def compare_runs(
        cls,
        run_dirs: List[Union[str, Path]],
        strict_compatibility: bool = False,
    ) -> Dict[str, Any]:
        runs_data = [cls.load_run(d) for d in run_dirs]

        compatibility = cls.check_compatibility(runs_data)
        if strict_compatibility and not compatibility["is_comparable"]:
            raise ValueError(
                f"Runs are not comparable: {'; '.join(compatibility['reasons'])}"
            )

        rows = []
        for r in runs_data:
            m = r["manifest"]
            cfg = m.get("config", {})
            metrics = m.get("final_metrics", {})
            eval_rep = r.get("eval_report", {}) or {}

            rows.append({
                "run_id": m.get("run_id", "unknown"),
                "task_type": m.get("task_type", "unknown"),
                "model": cfg.get("model", {}).get("model_name_or_path", "unknown"),
                "batch_size": cfg.get("hardware", {}).get("per_device_train_batch_size", 0),
                "grad_accum": cfg.get("hardware", {}).get("gradient_accumulation_steps", 0),
                "learning_rate": cfg.get("optimizer", {}).get("learning_rate", 0.0),
                "train_final_loss": metrics.get("final_loss", None),
                "heldout_eval_loss": eval_rep.get("eval_loss", None),
                "heldout_perplexity": eval_rep.get("perplexity", None),
                "dataset_fingerprint": eval_rep.get("dataset_fingerprint", "N/A"),
                "total_steps": metrics.get("total_steps", 0),
                "duration_sec": metrics.get("total_duration_sec", 0.0),
            })

        # Generate Markdown comparison table
        md_table = [
            "| Run ID | Task | Model | Batch x Accum | LR | Train Loss | Held-out Loss | Perplexity | Steps | Duration (s) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            h_loss_str = f"{row['heldout_eval_loss']}" if row['heldout_eval_loss'] is not None else "N/A"
            ppl_str = f"{row['heldout_perplexity']}" if row['heldout_perplexity'] is not None else "N/A"
            t_loss_str = f"{row['train_final_loss']}" if row['train_final_loss'] is not None else "N/A"

            md_table.append(
                f"| `{row['run_id']}` | {row['task_type']} | `{row['model']}` | "
                f"{row['batch_size']}x{row['grad_accum']} | {row['learning_rate']} | "
                f"{t_loss_str} | {h_loss_str} | {ppl_str} | "
                f"{row['total_steps']} | {row['duration_sec']} |"
            )

        if not compatibility["is_comparable"]:
            md_table.append("\n> [!WARNING] **Compatibility Notice**")
            for reason in compatibility["reasons"]:
                md_table.append(f"> - {reason}")

        return {
            "runs": rows,
            "compatibility": compatibility,
            "markdown_report": "\n".join(md_table),
        }
