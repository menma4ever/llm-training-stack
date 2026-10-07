"""Data and tokenizer preflight inspection routines.

Audits vocabulary alignment between models and tokenizers, inspects special token mappings,
analyzes sequence length token distributions, and provides isolated subprocess sandboxing
to prevent out-of-bounds CUDA crashes, loss masking corruption, and host memory exhaustion.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

from llm_training_stack.preflight.resource_identity import ResourceClassifier, ResourceKind


def _cleanup_process_tree(proc: subprocess.Popen, psutil_proc: Optional[psutil.Process], timeout_sec: float = 3.0):
    """Recursively terminates descendant processes, sends SIGKILL if necessary, and reaps zombies."""
    children = []
    if psutil_proc is not None:
        try:
            children = psutil_proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []

    for child in children:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        proc.terminate()
    except Exception:
        pass

    procs_to_wait = [p for p in children if p.is_running()]
    if psutil_proc is not None and psutil_proc.is_running():
        procs_to_wait.append(psutil_proc)

    if procs_to_wait:
        try:
            gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout_sec)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def _get_aggregate_rss_mb(p: psutil.Process) -> float:
    """Calculates total RSS memory in MB across process and all active descendants."""
    total_rss = 0
    try:
        total_rss += p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        for child in p.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return round(total_rss / (1024 ** 2), 2)


class TokenizerInspector:
    """Performs rigorous vocabulary audits and special token contract verification."""

    @staticmethod
    def audit_vocabulary(
        model_or_config: Any,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        """Audits vocabulary alignment between model architecture and tokenizer.

        Checks for index-out-of-bounds risk where tokenizer vocab exceeds model embedding matrix,
        which triggers fatal CUDA device-side assertions during forward passes.
        """
        # Extract model vocab size
        if hasattr(model_or_config, "config") and hasattr(model_or_config.config, "vocab_size"):
            model_vocab_size = model_or_config.config.vocab_size
        elif hasattr(model_or_config, "vocab_size"):
            model_vocab_size = model_or_config.vocab_size
        elif isinstance(model_or_config, dict) and "vocab_size" in model_or_config:
            model_vocab_size = model_or_config["vocab_size"]
        else:
            model_vocab_size = None

        # Extract tokenizer vocab size
        if hasattr(tokenizer, "vocab_size"):
            tokenizer_vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, "get_vocab"):
            tokenizer_vocab_size = len(tokenizer.get_vocab())
        elif hasattr(tokenizer, "__len__"):
            tokenizer_vocab_size = len(tokenizer)
        else:
            tokenizer_vocab_size = None

        tokenizer_len = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer_vocab_size

        warnings: List[str] = []
        is_safe = True
        verdict = "PASS_ALIGNED"

        if model_vocab_size is not None and tokenizer_len is not None:
            if tokenizer_len > model_vocab_size:
                is_safe = False
                verdict = "FAIL_EMBEDDING_OVERFLOW"
                warnings.append(
                    f"CRITICAL: Tokenizer vocab size ({tokenizer_len}) exceeds model vocab size "
                    f"({model_vocab_size}). Tokens with IDs >= {model_vocab_size} will cause fatal "
                    f"CUDA device-side assertions or index out-of-bounds in nn.Embedding."
                )
            elif model_vocab_size > tokenizer_len:
                verdict = "WARNING_UNMAPPED_EMBEDDINGS"
                warnings.append(
                    f"Model embedding layer ({model_vocab_size}) is larger than tokenizer vocab "
                    f"({tokenizer_len}). This is common for padded vocabularies (e.g., modulo 64/128 "
                    f"for tensor core efficiency), but unused embedding weights will remain unupdated."
                )

        return {
            "model_vocab_size": model_vocab_size,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "tokenizer_total_length": tokenizer_len,
            "vocab_difference": (tokenizer_len - model_vocab_size) if (tokenizer_len and model_vocab_size) else 0,
            "is_safe": is_safe,
            "verdict": verdict,
            "warnings": warnings,
        }

    @staticmethod
    def audit_special_tokens(tokenizer: Any) -> Dict[str, Any]:
        """Audits special token configuration: pad, bos, eos, unk, and collision gotchas."""
        special_tokens_map: Dict[str, Any] = {}
        tokens_to_check = ["bos_token", "eos_token", "pad_token", "unk_token"]

        for tok_name in tokens_to_check:
            tok_val = getattr(tokenizer, tok_name, None)
            tok_id = getattr(tokenizer, f"{tok_name}_id", None)
            special_tokens_map[tok_name] = {
                "token": tok_val,
                "id": tok_id,
                "is_defined": tok_val is not None and tok_id is not None,
            }

        warnings: List[str] = []
        pad_defined = special_tokens_map["pad_token"]["is_defined"]
        pad_id = special_tokens_map["pad_token"]["id"]
        eos_id = special_tokens_map["eos_token"]["id"]
        pad_equals_eos = (pad_id is not None and eos_id is not None and pad_id == eos_id)

        if not pad_defined:
            warnings.append(
                "CRITICAL: pad_token is NOT defined. Batched training or collation will fail or throw errors. "
                "Assign tokenizer.pad_token = tokenizer.eos_token or add a dedicated [PAD] token."
            )

        if pad_equals_eos:
            warnings.append(
                "NOTICE: pad_token_id is identical to eos_token_id. Standard practice for Llama/Qwen, "
                "but requires explicit attention masking and DataCollatorForCompletionOnlyLM / loss masking "
                "to prevent model from learning to ignore real end-of-sequence boundaries."
            )

        # Check for chat template tokens if present
        chat_template = getattr(tokenizer, "chat_template", None)

        verdict = "PASS_SAFE"
        if not pad_defined:
            verdict = "FAIL_MISSING_PAD_TOKEN"
        elif pad_equals_eos:
            verdict = "PASS_WITH_COLLISION_NOTICE"

        return {
            "special_tokens": special_tokens_map,
            "pad_token_defined": pad_defined,
            "pad_equals_eos": pad_equals_eos,
            "has_chat_template": chat_template is not None,
            "verdict": verdict,
            "warnings": warnings,
        }


class DataInspector:
    """Analyzes dataset token length distributions, truncation risk, and padding overhead."""

    @staticmethod
    def analyze_token_distribution(
        tokenizer: Any,
        samples: List[Union[str, List[int]]],
        max_seq_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates token distribution metrics, percentiles, truncation count, and padding ratio."""
        if not samples:
            return {
                "total_samples": 0,
                "verdict": "ERROR_EMPTY_DATASET",
                "warnings": ["No samples provided for analysis."],
            }

        lengths: List[int] = []
        for item in samples:
            if isinstance(item, str):
                encoded = tokenizer.encode(item, add_special_tokens=False) if hasattr(tokenizer, "encode") else item.split()
                lengths.append(len(encoded))
            elif isinstance(item, list):
                lengths.append(len(item))
            elif hasattr(item, "__len__"):
                lengths.append(len(item))
            else:
                lengths.append(1)

        lengths.sort()
        n = len(lengths)

        def percentile(p: float) -> int:
            idx = int(math.ceil((p / 100.0) * n)) - 1
            idx = max(0, min(idx, n - 1))
            return lengths[idx]

        min_len = lengths[0]
        max_len = lengths[-1]
        mean_len = round(sum(lengths) / n, 2)
        median_len = percentile(50)
        p75 = percentile(75)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)

        truncated_count = 0
        truncated_percent = 0.0
        padding_ratio_percent = 0.0
        effective_tokens = 0
        total_tokens = sum(lengths)

        warnings: List[str] = []
        if min_len == 0:
            warnings.append("WARNING: Dataset contains empty sequences (length 0).")

        if max_seq_length is not None and max_seq_length > 0:
            truncated_count = sum(1 for l in lengths if l > max_seq_length)
            truncated_percent = round((truncated_count / n) * 100, 2)
            effective_tokens = sum(min(l, max_seq_length) for l in lengths)
            theoretical_max_capacity = n * max_seq_length
            padding_tokens = theoretical_max_capacity - effective_tokens
            padding_ratio_percent = round((padding_tokens / theoretical_max_capacity) * 100, 2)

            if truncated_percent > 15.0:
                warnings.append(
                    f"WARNING: High truncation rate: {truncated_percent}% of samples ({truncated_count}/{n}) "
                    f"exceed max_seq_length={max_seq_length} and will lose trailing context."
                )
            if padding_ratio_percent > 60.0:
                warnings.append(
                    f"NOTICE: High padding overhead: ~{padding_ratio_percent}% of allocated sequence capacity "
                    f"will be padding tokens. Consider dynamic padding or sample packing."
                )

        verdict = "PASS_CLEAN_DISTRIBUTION"
        if truncated_percent > 25.0:
            verdict = "WARNING_HIGH_TRUNCATION"
        elif min_len == 0:
            verdict = "WARNING_CONTAINS_EMPTY_SAMPLES"

        return {
            "total_samples": n,
            "min_length": min_len,
            "max_length": max_len,
            "mean_length": mean_len,
            "median_length": median_len,
            "p50": median_len,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "target_max_seq_length": max_seq_length,
            "truncated_count": truncated_count,
            "truncated_percent": truncated_percent,
            "total_tokens": total_tokens,
            "effective_tokens": effective_tokens if max_seq_length else total_tokens,
            "estimated_padding_overhead_percent": padding_ratio_percent if max_seq_length else 0.0,
            "verdict": verdict,
            "warnings": warnings,
        }


class PreflightInspectionSuite:
    """Unified preflight suite aggregating vocabulary, special tokens, and distribution audits."""

    @staticmethod
    def run_full_inspection(
        model_or_config: Any,
        tokenizer: Any,
        samples: Optional[List[Union[str, List[int]]]] = None,
        max_seq_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs complete tokenizer and dataset preflight checks."""
        vocab_report = TokenizerInspector.audit_vocabulary(model_or_config, tokenizer)
        special_tokens_report = TokenizerInspector.audit_special_tokens(tokenizer)

        dist_report = None
        if samples is not None and len(samples) > 0:
            dist_report = DataInspector.analyze_token_distribution(
                tokenizer=tokenizer,
                samples=samples,
                max_seq_length=max_seq_length,
            )

        all_warnings: List[str] = []
        all_warnings.extend(vocab_report.get("warnings", []))
        all_warnings.extend(special_tokens_report.get("warnings", []))
        if dist_report:
            all_warnings.extend(dist_report.get("warnings", []))

        # Overall health verdict
        if not vocab_report.get("is_safe", True) or not special_tokens_report.get("pad_token_defined", True):
            overall_status = "CRITICAL_ISSUES_FOUND"
        elif all_warnings:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "PASSED_CLEAN"

        return {
            "overall_status": overall_status,
            "vocabulary_audit": vocab_report,
            "special_tokens_audit": special_tokens_report,
            "distribution_audit": dist_report,
            "consolidated_warnings": all_warnings,
        }

    @classmethod
    def run_isolated_inspection(
        cls,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        samples: Optional[List[str]] = None,
        max_seq_length: Optional[int] = None,
        max_process_memory_mb: int = 2048,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """Runs preflight tokenizer and model inspection inside a bounded isolated subprocess."""
        target_tok = tokenizer_path or model_path

        # Classify resource identities
        model_identity = ResourceClassifier.classify(model_path)
        tok_identity = ResourceClassifier.classify(target_tok)

        if model_identity.is_remote or tok_identity.is_remote:
            return {
                "overall_status": "CRITICAL_ISSUES_FOUND",
                "verdict": "SECURITY_POLICY_VIOLATION",
                "error_detail": "Remote URI protocols are disallowed in inspection.",
            }

        # Subprocess script
        code = f"""
import json, sys
from transformers import AutoConfig, AutoTokenizer
from llm_training_stack.preflight.data_inspector import PreflightInspectionSuite

try:
    tok = AutoTokenizer.from_pretrained({repr(target_tok)})
    cfg = AutoConfig.from_pretrained({repr(model_path)})
    samples = {repr(samples)}
    max_seq_len = {repr(max_seq_length)}
    res = PreflightInspectionSuite.run_full_inspection(cfg, tok, samples=samples, max_seq_length=max_seq_len)
    print("INSPECT_JSON_START")
    print(json.dumps(res))
    print("INSPECT_JSON_END")
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
"""
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parent.parent.parent
        env["PYTHONPATH"] = f"{str(project_root)}{os.pathsep}{env.get('PYTHONPATH', '')}"

        start_time = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        from llm_training_stack.preflight.probe import BoundedStreamDrainer
        stdout_drainer = BoundedStreamDrainer(proc.stdout, max_bytes=32768)
        stderr_drainer = BoundedStreamDrainer(proc.stderr, max_bytes=32768)

        try:
            psutil_proc = psutil.Process(proc.pid)
        except Exception:
            psutil_proc = None

        timed_out = False
        memory_exceeded = False
        peak_rss_mb = 0.0

        while proc.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                timed_out = True
                _cleanup_process_tree(proc, psutil_proc)
                break

            if psutil_proc is not None:
                current_rss = _get_aggregate_rss_mb(psutil_proc)
                peak_rss_mb = max(peak_rss_mb, current_rss)
                if current_rss > max_process_memory_mb:
                    memory_exceeded = True
                    _cleanup_process_tree(proc, psutil_proc)
                    break

            time.sleep(0.05)

        try:
            proc.wait(timeout=3)
        except Exception:
            _cleanup_process_tree(proc, psutil_proc)

        stdout = stdout_drainer.get_output(timeout_sec=2.0)
        stderr = stderr_drainer.get_output(timeout_sec=2.0)

        if timed_out:
            return {
                "overall_status": "CRITICAL_ISSUES_FOUND",
                "verdict": "INSPECTION_TIMEOUT",
                "measured_peak_memory_mb": peak_rss_mb,
                "error_detail": f"Tokenizer inspection timed out after {timeout_seconds}s.",
            }

        if memory_exceeded:
            return {
                "overall_status": "CRITICAL_ISSUES_FOUND",
                "verdict": "PROCESS_MEMORY_LIMIT_EXCEEDED",
                "measured_peak_memory_mb": peak_rss_mb,
                "error_detail": f"Inspection exceeded process memory budget ({peak_rss_mb} MB > {max_process_memory_mb} MB).",
            }

        if "INSPECT_JSON_START" in stdout and "INSPECT_JSON_END" in stdout:
            try:
                raw_json = stdout.split("INSPECT_JSON_START")[1].split("INSPECT_JSON_END")[0].strip()
                result = json.loads(raw_json)
                result["measured_peak_memory_mb"] = peak_rss_mb
                return result
            except Exception as parse_err:
                stderr += f"\nParse error: {parse_err}"

        error_msg = stderr.strip() or stdout.strip() or "Unknown inspection error"
        is_oom = "out of memory" in error_msg.lower() or proc.returncode in [-9, 137]
        return {
            "overall_status": "CRITICAL_ISSUES_FOUND",
            "verdict": "OOM_PREVENTED" if is_oom else "INSPECTION_FAILED",
            "measured_peak_memory_mb": peak_rss_mb,
            "error_detail": error_msg[:4096],
        }
