"""Tests for checkpoint bundling, atomic persistence, and retention management."""

import torch
import torch.nn as nn
from llm_training_stack.checkpoints.manager import CheckpointManager
from llm_training_stack.checkpoints.resume import ResumeManager


def test_checkpoint_save_and_inspect(tmp_path, micro_llama_model):
    cp_manager = CheckpointManager(tmp_path, save_total_limit=2)
    optimizer = torch.optim.AdamW(micro_llama_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=10)

    # Save checkpoint at step 10
    cp_path = cp_manager.save_checkpoint(
        step=10,
        epoch=1.0,
        model=micro_llama_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss=1.85,
        dataloader_index=10,
        is_peft=False,
    )

    assert cp_path.exists()
    assert (cp_path / "model.safetensors").exists()
    assert (cp_path / "optimizer.pt").exists()
    assert (cp_path / "scheduler.pt").exists()
    assert (cp_path / "rng_state.pt").exists()
    assert (cp_path / "metadata.json").exists()

    info = ResumeManager.inspect_checkpoint(cp_path)
    assert info["step"] == 10
    assert info["loss"] == 1.85
    assert info["is_complete_bundle"] is True


def test_checkpoint_rotation(tmp_path, micro_llama_model):
    cp_manager = CheckpointManager(tmp_path, save_total_limit=2)
    optimizer = torch.optim.AdamW(micro_llama_model.parameters(), lr=1e-4)

    for step in [10, 20, 30]:
        cp_manager.save_checkpoint(
            step=step,
            epoch=step / 10.0,
            model=micro_llama_model,
            optimizer=optimizer,
            lr_scheduler=None,
            loss=2.0 - step * 0.01,
        )

    # With save_total_limit=2, checkpoint-10 should be pruned, leaving 20 and 30
    assert not (tmp_path / "checkpoint-10").exists()
    assert (tmp_path / "checkpoint-20").exists()
    assert (tmp_path / "checkpoint-30").exists()
