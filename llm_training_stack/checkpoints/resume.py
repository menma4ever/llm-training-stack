"""Checkpoint resumption and state verification."""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_model


class ResumeManager:
    """Restores full training state with deterministic RNG and optimizer alignment."""

    @staticmethod
    def inspect_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
        cp_path = Path(checkpoint_dir)
        if not cp_path.exists() or not cp_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {cp_path}")

        meta_file = cp_path / "metadata.json"
        metadata = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        has_safetensors = (cp_path / "model.safetensors").exists() or (cp_path / "adapter_model.safetensors").exists()
        has_optimizer = (cp_path / "optimizer.pt").exists()
        has_scheduler = (cp_path / "scheduler.pt").exists()
        has_rng = (cp_path / "rng_state.pt").exists()
        has_ref_identity = (cp_path / "ref_model_identity.json").exists() or (cp_path.parent / "ref_model_identity.json").exists()

        is_valid = has_safetensors and has_optimizer and has_rng

        return {
            "path": str(cp_path),
            "step": metadata.get("step", 0),
            "epoch": metadata.get("epoch", 0.0),
            "loss": metadata.get("loss", None),
            "dataloader_index": metadata.get("dataloader_index", 0),
            "is_peft": metadata.get("is_peft", False),
            "has_weights": has_safetensors,
            "has_optimizer": has_optimizer,
            "has_scheduler": has_scheduler,
            "has_rng": has_rng,
            "has_ref_identity": has_ref_identity,
            "is_complete_bundle": is_valid,
        }

    @classmethod
    def resume_into(
        cls,
        checkpoint_dir: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        cp_path = Path(checkpoint_dir)
        inspection = cls.inspect_checkpoint(cp_path)

        if not inspection["is_complete_bundle"]:
            raise ValueError(f"Incomplete checkpoint bundle in {cp_path}: {inspection}")

        # 0. Validate Reference Model if supplied (e.g. DPO full-parameter tuning)
        if ref_model is not None:
            ref_identity_file = cp_path / "ref_model_identity.json"
            if not ref_identity_file.exists():
                ref_identity_file = cp_path.parent / "ref_model_identity.json"
            if not ref_identity_file.exists():
                raise FileNotFoundError(
                    f"Missing required ref_model_identity.json in checkpoint {cp_path} "
                    f"for reference model validation."
                )
            with open(ref_identity_file, "r", encoding="utf-8") as f:
                ref_info = json.load(f)
            num_p = sum(p.numel() for p in ref_model.parameters())
            if "num_parameters" in ref_info and ref_info["num_parameters"] != num_p:
                raise ValueError(
                    f"Reference model parameter count mismatch on resume: "
                    f"expected {ref_info['num_parameters']}, got {num_p}."
                )
            if "param_hash" in ref_info:
                from llm_training_stack.pipelines.dpo import compute_model_weight_hash
                curr_hash = compute_model_weight_hash(ref_model)
                if ref_info["param_hash"] != curr_hash:
                    raise ValueError(
                        f"Reference model parameter integrity/weight hash mismatch on resume: "
                        f"expected {ref_info['param_hash']}, got {curr_hash}."
                    )
            if "revision" in ref_info and ref_info["revision"] is not None:
                model_cfg = getattr(ref_model, "config", None)
                model_rev = getattr(model_cfg, "_commit_hash", getattr(model_cfg, "revision", None))
                if model_rev is not None and model_rev != ref_info["revision"]:
                    raise ValueError(
                        f"Reference model revision mismatch on resume: "
                        f"expected {ref_info['revision']}, got {model_rev}."
                    )

        # 1. Restore Model / Adapter weights
        if inspection["is_peft"]:
            if hasattr(model, "load_adapter"):
                model.load_adapter(str(cp_path), adapter_name="default")
            elif hasattr(model, "from_pretrained"):
                pass  # Model loaded directly via PeftModel
        else:
            model_safetensors = cp_path / "model.safetensors"
            if model_safetensors.exists():
                load_model(model, str(model_safetensors))

        # 2. Restore Optimizer
        if optimizer is not None and (cp_path / "optimizer.pt").exists():
            opt_state = torch.load(cp_path / "optimizer.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_state)

        # 3. Restore Scheduler
        if lr_scheduler is not None and (cp_path / "scheduler.pt").exists():
            sched_state = torch.load(cp_path / "scheduler.pt", map_location="cpu", weights_only=False)
            lr_scheduler.load_state_dict(sched_state)

        # 4. Restore RNG State
        if (cp_path / "rng_state.pt").exists():
            rng_states = torch.load(cp_path / "rng_state.pt", map_location="cpu", weights_only=False)
            if "torch_cpu" in rng_states:
                torch.set_rng_state(rng_states["torch_cpu"])
            if "numpy" in rng_states:
                np.random.set_state(rng_states["numpy"])
            if "python_random" in rng_states:
                random.setstate(rng_states["python_random"])
            if "torch_cuda" in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])

        return inspection
