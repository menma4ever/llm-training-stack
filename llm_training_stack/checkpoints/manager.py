"""Comprehensive checkpoint manager persisting model, optimizer, scheduler, RNG and data states."""

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_model


class CheckpointManager:
    """Saves atomic checkpoint bundles guaranteeing full training reproducibility."""

    def __init__(self, output_dir: Path, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        epoch: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any],
        loss: float,
        dataloader_index: int = 0,
        is_peft: bool = False,
    ) -> Path:
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Model / Adapter weights
        if is_peft and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(checkpoint_dir))
        else:
            # Use safetensors for full model weights
            if hasattr(model, "state_dict"):
                save_model(model, str(checkpoint_dir / "model.safetensors"))
            else:
                torch.save(model, checkpoint_dir / "model.pt")

        # 2. Save Optimizer State
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # 3. Save Scheduler State
        if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
            torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # 4. Save RNG States
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python_random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng_state.pt")

        # 5. Save Metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "dataloader_index": dataloader_index,
            "is_peft": is_peft,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5.1 Preserve ref_model_identity.json in checkpoint bundle if present
        ref_id_path = self.output_dir / "ref_model_identity.json"
        if ref_id_path.exists():
            shutil.copy2(ref_id_path, checkpoint_dir / "ref_model_identity.json")

        # 6. Rotate checkpoints to enforce save_total_limit
        self._rotate_checkpoints()

        return checkpoint_dir

    def _rotate_checkpoints(self) -> None:
        if self.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0
        )

        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
