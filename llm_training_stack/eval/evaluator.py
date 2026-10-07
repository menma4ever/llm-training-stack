"""Held-out dataset evaluation and perplexity calculation with response-only masking."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_dataset_fingerprint(samples: List[Any]) -> str:
    """Calculates deterministic SHA256 digest of evaluation sample content with framed boundaries."""
    hasher = hashlib.sha256()
    for s in samples:
        if isinstance(s, dict):
            serialized = json.dumps(s, sort_keys=True).encode("utf-8")
        else:
            serialized = str(s).encode("utf-8")
        # Length-prefixed framing eliminates boundary collisions (e.g. ['ab','c'] vs ['a','bc'])
        hasher.update(f"{len(serialized)}:".encode("utf-8"))
        hasher.update(serialized)
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


class Evaluator:
    """Evaluates cross-entropy loss and perplexity on held-out test splits."""

    @staticmethod
    def evaluate(
        model: nn.Module,
        tokenizer: AutoTokenizer,
        eval_samples: List[Union[str, Dict[str, Any]]],
        max_seq_length: int = 1024,
        device: Optional[torch.device] = None,
        model_name_or_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        total_loss = 0.0
        total_tokens = 0
        valid_samples_count = 0
        used_response_masking = False

        # Validate that eval_samples contains at least one non-empty string or dict
        has_non_empty = False
        for s in eval_samples:
            if isinstance(s, str) and s.strip():
                has_non_empty = True
                break
            elif isinstance(s, dict) and any(str(v).strip() for v in s.values()):
                has_non_empty = True
                break

        if not has_non_empty:
            raise ValueError("Evaluation dataset contains no non-empty text samples")

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        with torch.no_grad():
            for sample in eval_samples:
                if isinstance(sample, str):
                    text = sample.strip()
                    if not text:
                        continue
                    enc = tokenizer(
                        text,
                        max_length=max_seq_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    input_ids = enc["input_ids"]
                    labels = input_ids.clone()

                elif isinstance(sample, dict):
                    prompt_val = sample.get("prompt")
                    response_val = sample.get("response")

                    if prompt_val is not None and response_val is not None:
                        # Instruction / dialogue evaluation: compute loss STRICTLY on response tokens
                        prompt_str = str(prompt_val)
                        response_str = str(response_val)
                        if not prompt_str.strip() and not response_str.strip():
                            continue

                        # Tokenize prompt to determine prompt prefix token length
                        prompt_enc = tokenizer(
                            prompt_str,
                            max_length=max_seq_length,
                            truncation=True,
                            add_special_tokens=True,
                        )
                        prompt_token_len = len(prompt_enc["input_ids"])

                        # Tokenize combined prompt + response
                        full_text = prompt_str + response_str
                        enc = tokenizer(
                            full_text,
                            max_length=max_seq_length,
                            truncation=True,
                            return_tensors="pt",
                        ).to(device)
                        input_ids = enc["input_ids"]
                        labels = input_ids.clone()

                        # Mask prompt tokens with -100 so ONLY response tokens contribute to eval loss
                        mask_len = min(prompt_token_len, input_ids.size(1))
                        labels[:, :mask_len] = -100
                        used_response_masking = True
                    else:
                        text = sample.get("text", "")
                        if not str(text).strip():
                            continue
                        enc = tokenizer(
                            str(text),
                            max_length=max_seq_length,
                            truncation=True,
                            return_tensors="pt",
                        ).to(device)
                        input_ids = enc["input_ids"]
                        labels = input_ids.clone()
                else:
                    continue

                if input_ids.size(1) < 2:
                    continue

                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") and outputs.logits is not None else outputs[0]

                # Shift by 1 token for causal language modeling prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                valid_mask = (shift_labels.view(-1) != -100)
                valid_tokens = valid_mask.sum().item()

                if valid_tokens > 0:
                    total_loss += per_token_loss[valid_mask].sum().item()
                    total_tokens += valid_tokens
                    valid_samples_count += 1

        if valid_samples_count == 0 or total_tokens == 0:
            raise ValueError("Evaluation dataset contains no non-empty text samples")

        avg_loss = total_loss / total_tokens
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        resolved_model = model_name_or_path or getattr(getattr(model, "config", None), "_name_or_path", "model")
        resolved_tok = getattr(tokenizer, "name_or_path", "tokenizer")

        return {
            "eval_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 3),
            "evaluated_samples": valid_samples_count,
            "total_tokens": total_tokens,
            "dataset_fingerprint": compute_dataset_fingerprint(eval_samples),
            "tokenizer_name_or_path": str(resolved_tok),
            "model_name_or_path": str(resolved_model),
            "masking_scheme": "response_only" if used_response_masking else "all_tokens",
            "max_seq_length": int(max_seq_length),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def save_report(report: Dict[str, Any], path: Union[str, Path]) -> None:
        """Persists evaluation report contract to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    @staticmethod
    def load_report(path: Union[str, Path]) -> Dict[str, Any]:
        """Loads persisted evaluation report contract from disk."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
