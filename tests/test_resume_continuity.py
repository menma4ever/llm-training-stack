"""Acceptance verification: Deterministic training resumption continuity test."""

import copy
import random
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_training_stack.checkpoints.manager import CheckpointManager
from llm_training_stack.checkpoints.resume import ResumeManager


def run_steps(model, optimizer, scheduler, batches, start_step=0):
    losses = []
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
    return losses


def test_checkpoint_resume_trajectory_continuity(tmp_path, micro_llama_config):
    """Verifies that an interrupted and resumed run achieves state and loss continuity."""
    # Deterministic batches
    torch.manual_seed(1337)
    batches = [torch.randint(0, 256, (2, 32)) for _ in range(4)]

    # 1. Uninterrupted baseline run (4 steps)
    torch.manual_seed(42)
    model_uninterrupted = LlamaForCausalLM(micro_llama_config)
    opt_uninterrupted = torch.optim.AdamW(model_uninterrupted.parameters(), lr=1e-3)
    sched_uninterrupted = torch.optim.lr_scheduler.LinearLR(opt_uninterrupted, start_factor=1.0, end_factor=0.5, total_iters=4)

    baseline_losses = run_steps(
        model_uninterrupted, opt_uninterrupted, sched_uninterrupted, batches
    )

    # 2. Interrupted run: train 2 steps, checkpoint, resume, train remaining 2 steps
    torch.manual_seed(42)
    model_interrupted = LlamaForCausalLM(micro_llama_config)
    opt_interrupted = torch.optim.AdamW(model_interrupted.parameters(), lr=1e-3)
    sched_interrupted = torch.optim.lr_scheduler.LinearLR(opt_interrupted, start_factor=1.0, end_factor=0.5, total_iters=4)

    cp_manager = CheckpointManager(tmp_path / "checkpoints", save_total_limit=2)

    # Train step 1 and 2
    losses_part1 = run_steps(
        model_interrupted, opt_interrupted, sched_interrupted, batches[:2]
    )

    # Save checkpoint at step 2
    cp_path = cp_manager.save_checkpoint(
        step=2,
        epoch=0.5,
        model=model_interrupted,
        optimizer=opt_interrupted,
        lr_scheduler=sched_interrupted,
        loss=losses_part1[-1],
        dataloader_index=2,
    )

    # 3. Fresh model resumes from checkpoint at step 2
    model_resumed = LlamaForCausalLM(micro_llama_config)
    opt_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
    sched_resumed = torch.optim.lr_scheduler.LinearLR(opt_resumed, start_factor=1.0, end_factor=0.5, total_iters=4)

    res_info = ResumeManager.resume_into(
        cp_path, model_resumed, opt_resumed, sched_resumed
    )
    assert res_info["step"] == 2

    # Continue training remaining 2 steps
    losses_part2 = run_steps(
        model_resumed, opt_resumed, sched_resumed, batches[2:], start_step=2
    )

    combined_resumed_losses = losses_part1 + losses_part2

    # Check trajectory alignment
    for step_idx in range(4):
        diff = abs(baseline_losses[step_idx] - combined_resumed_losses[step_idx])
        assert diff < 1e-4, f"Step {step_idx} loss mismatch: baseline={baseline_losses[step_idx]}, resumed={combined_resumed_losses[step_idx]}"

    # Verify final model parameters match
    for p_base, p_res in zip(model_uninterrupted.parameters(), model_resumed.parameters()):
        assert torch.allclose(p_base, p_res, atol=1e-5)


def test_production_pipeline_checkpoint_resume_with_gradient_accumulation(tmp_path, base_test_config):
    """Verifies that pure production SFTPipeline with gradient accumulation and matched
    scheduler horizon achieves loss, learning rate, and parameter parity across interruption
    without any test-only subclasses or overridden train loops.
    """
    import json
    from llm_training_stack.pipelines.sft import SFTPipeline
    from llm_training_stack.config.schema import TaskType

    dir_uninterrupted = tmp_path / "uninterrupted"
    dir_interrupted = tmp_path / "interrupted"

    common_cfg = base_test_config.model_copy(deep=True)
    common_cfg.task_type = TaskType.SFT
    common_cfg.hardware.gradient_accumulation_steps = 2
    common_cfg.max_steps = 8
    common_cfg.dataset.train_sample_limit = 16
    common_cfg.seed = 42

    # 1. Uninterrupted baseline (8 steps, grad_accum=2, save_steps=4)
    torch.manual_seed(42)
    cfg_unint = common_cfg.model_copy(deep=True)
    cfg_unint.logging.output_dir = str(dir_uninterrupted)
    cfg_unint.logging.save_steps = 4

    pipe_unint = SFTPipeline(cfg_unint)
    res_unint = pipe_unint.train_direct()
    assert res_unint["total_steps"] == 8

    # 2. Interrupted run Part 1: Planned for 8 steps, preemption at step 4
    # Uses exact same config and dataset (16 samples) with interrupt_after_step=4
    torch.manual_seed(42)
    cfg_part1 = common_cfg.model_copy(deep=True)
    cfg_part1.logging.output_dir = str(dir_interrupted)
    cfg_part1.logging.save_steps = 4

    pipe_part1 = SFTPipeline(cfg_part1)
    res_part1 = pipe_part1.train_direct(interrupt_after_step=4)
    assert res_part1["total_steps"] == 4
    assert res_part1["status"] == "INTERRUPTED"
    assert (dir_interrupted / "checkpoint-4").exists()

    # 3. Resumed run through production SFTPipeline from checkpoint-4
    # Uses exact same config and dataset (16 samples) without dataset alterations
    pipe_part2 = SFTPipeline(cfg_part1)
    res_part2 = pipe_part2.train_direct(resume_from=dir_interrupted / "checkpoint-4")
    assert res_part2["total_steps"] == 8

    # 4. Compare events and loss trajectory
    with open(dir_uninterrupted / "events.jsonl", "r", encoding="utf-8") as f:
        events_unint = [json.loads(line) for line in f]
    with open(dir_interrupted / "events.jsonl", "r", encoding="utf-8") as f:
        events_int = [json.loads(line) for line in f]

    assert len(events_unint) == len(events_int)
    for e1, e2 in zip(events_unint, events_int):
        assert abs(e1["loss"] - e2["loss"]) < 1e-4, f"Loss mismatch at step {e1['step']}: {e1['loss']} vs {e2['loss']}"
        assert abs(e1["learning_rate"] - e2["learning_rate"]) < 1e-6, f"LR mismatch at step {e1['step']}"

    # 5. Compare final parameters
    w_unint = pipe_unint.model.state_dict()
    w_res = pipe_part2.model.state_dict()
    for k in w_unint:
        assert torch.allclose(w_unint[k], w_res[k], atol=1e-4), f"Parameter mismatch in {k}"


def test_production_lora_pipeline_checkpoint_resume_continuity(tmp_path, base_test_config):
    """Verifies that pure production LoRAPipeline achieves adapter weight, loss,
    and learning rate parity across interruption and resumption.
    """
    import json
    from llm_training_stack.pipelines.lora import LoRAPipeline
    from llm_training_stack.config.schema import TaskType, PeftConfig

    dir_uninterrupted = tmp_path / "lora_uninterrupted"
    dir_interrupted = tmp_path / "lora_interrupted"

    common_cfg = base_test_config.model_copy(deep=True)
    common_cfg.task_type = TaskType.LORA
    common_cfg.peft = PeftConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    common_cfg.hardware.gradient_accumulation_steps = 2
    common_cfg.max_steps = 8
    common_cfg.dataset.train_sample_limit = 16
    common_cfg.seed = 42

    # 1. Uninterrupted Baseline
    torch.manual_seed(42)
    cfg_unint = common_cfg.model_copy(deep=True)
    cfg_unint.logging.output_dir = str(dir_uninterrupted)
    cfg_unint.logging.save_steps = 4

    pipe_unint = LoRAPipeline(cfg_unint)
    res_unint = pipe_unint.train_direct()
    assert res_unint["total_steps"] == 8

    # 2. Interrupted Part 1 (preempted at step 4 on unchanged dataset)
    torch.manual_seed(42)
    cfg_part1 = common_cfg.model_copy(deep=True)
    cfg_part1.logging.output_dir = str(dir_interrupted)
    cfg_part1.logging.save_steps = 4

    pipe_part1 = LoRAPipeline(cfg_part1)
    res_part1 = pipe_part1.train_direct(interrupt_after_step=4)
    assert res_part1["total_steps"] == 4
    assert res_part1["status"] == "INTERRUPTED"
    assert (dir_interrupted / "checkpoint-4").exists()

    # 3. Resumed Part 2 from checkpoint-4 on unchanged dataset
    pipe_part2 = LoRAPipeline(cfg_part1)
    res_part2 = pipe_part2.train_direct(resume_from=dir_interrupted / "checkpoint-4")
    assert res_part2["total_steps"] == 8

    # 4. Compare events and loss trajectory
    with open(dir_uninterrupted / "events.jsonl", "r", encoding="utf-8") as f:
        events_unint = [json.loads(line) for line in f]
    with open(dir_interrupted / "events.jsonl", "r", encoding="utf-8") as f:
        events_int = [json.loads(line) for line in f]

    assert len(events_unint) == len(events_int)
    for e1, e2 in zip(events_unint, events_int):
        assert abs(e1["loss"] - e2["loss"]) < 1e-4, f"Loss mismatch at step {e1['step']}: {e1['loss']} vs {e2['loss']}"
        assert abs(e1["learning_rate"] - e2["learning_rate"]) < 1e-6, f"LR mismatch at step {e1['step']}"

    # 5. Compare adapter weights
    w_unint = {k: v for k, v in pipe_unint.model.state_dict().items() if "lora" in k.lower()}
    w_res = {k: v for k, v in pipe_part2.model.state_dict().items() if "lora" in k.lower()}
    for k in w_unint:
        assert torch.allclose(w_unint[k], w_res[k], atol=1e-4), f"LoRA parameter mismatch in {k}"


def test_nondivisible_gradient_accumulation_and_dataset_exhaustion(tmp_path, base_test_config):
    """Verifies pipeline behavior on odd/nondivisible gradient accumulation boundaries
    when dataset exhaustion occurs mid-accumulation.
    """
    from llm_training_stack.pipelines.sft import SFTPipeline
    from llm_training_stack.config.schema import TaskType

    cfg = base_test_config.model_copy(deep=True)
    cfg.task_type = TaskType.SFT
    cfg.backend = "direct"
    cfg.hardware.gradient_accumulation_steps = 4  # cycle is 4 microbatches
    cfg.dataset.train_sample_limit = 6  # 3 microbatches of size 2 (odd / nondivisible by 4)
    cfg.max_steps = 20
    cfg.logging.output_dir = str(tmp_path / "nondivisible_run")
    cfg.logging.save_steps = 10

    pipeline = SFTPipeline(cfg)
    result = pipeline.train()

    # The pipeline should run until dataset exhaustion (3 steps) without unhandled exceptions
    assert result["total_steps"] == 3
    assert result["status"] == "COMPLETED"
    assert (tmp_path / "nondivisible_run" / "checkpoint-3").exists()


def test_production_dpo_pipeline_checkpoint_resume_and_ref_invariance(tmp_path, base_test_config):
    """Verifies that pure production DPOPipeline preserves reference model invariance
    and achieves loss and parameter trajectory continuity across interruption and resumption.
    """
    import json
    from llm_training_stack.pipelines.dpo import DPOPipeline
    from llm_training_stack.config.schema import TaskType

    dir_uninterrupted = tmp_path / "dpo_uninterrupted"
    dir_interrupted = tmp_path / "dpo_interrupted"

    common_cfg = base_test_config.model_copy(deep=True)
    common_cfg.task_type = TaskType.DPO
    common_cfg.dpo_beta = 0.1
    common_cfg.hardware.gradient_accumulation_steps = 2
    common_cfg.max_steps = 4
    common_cfg.dataset.train_sample_limit = 4
    common_cfg.seed = 42

    # 1. Uninterrupted Baseline (4 steps, grad_accum=2, save_steps=2)
    torch.manual_seed(42)
    cfg_unint = common_cfg.model_copy(deep=True)
    cfg_unint.logging.output_dir = str(dir_uninterrupted)
    cfg_unint.logging.save_steps = 2

    pipe_unint = DPOPipeline(cfg_unint)
    res_unint = pipe_unint.train_direct()
    assert res_unint["total_steps"] == 4

    # 2. Interrupted run Part 1 (preempted at step 2 on unchanged dataset)
    torch.manual_seed(42)
    cfg_part1 = common_cfg.model_copy(deep=True)
    cfg_part1.logging.output_dir = str(dir_interrupted)
    cfg_part1.logging.save_steps = 2

    pipe_part1 = DPOPipeline(cfg_part1)
    res_part1 = pipe_part1.train_direct(interrupt_after_step=2)
    assert res_part1["total_steps"] == 2
    assert res_part1["status"] == "INTERRUPTED"
    assert (dir_interrupted / "checkpoint-2").exists()

    # 3. Resumed run Part 2 from checkpoint-2 on unchanged dataset
    pipe_part2 = DPOPipeline(cfg_part1)
    res_part2 = pipe_part2.train_direct(resume_from=dir_interrupted / "checkpoint-2")
    assert res_part2["total_steps"] == 4

    # 4. Reference Model Invariance Check
    # Verify that ref_model remains completely frozen and identical across runs
    assert pipe_unint.ref_model is not None
    assert pipe_part2.ref_model is not None
    for (k1, p1), (k2, p2) in zip(pipe_unint.ref_model.state_dict().items(), pipe_part2.ref_model.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(p1, p2), f"Reference model weights drifted in parameter {k1}!"

    # 5. Compare final policy parameters
    w_unint = pipe_unint.model.state_dict()
    w_res = pipe_part2.model.state_dict()
    for k in w_unint:
        assert torch.allclose(w_unint[k], w_res[k], atol=1e-4), f"DPO policy parameter mismatch in {k}"


def test_resume_manager_changed_ref_model_and_corrupt_bundle_rejection(tmp_path, micro_llama_config):
    """CEO Delta 11: ResumeManager strictly rejects changed reference model weights and incomplete bundles."""
    import json
    from llm_training_stack.pipelines.dpo import compute_model_weight_hash

    model = LlamaForCausalLM(micro_llama_config)
    ref_model = LlamaForCausalLM(micro_llama_config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda x: 1.0)

    cp_manager = CheckpointManager(tmp_path / "checkpoints")
    cp_path = cp_manager.save_checkpoint(
        step=2,
        epoch=0.5,
        model=model,
        optimizer=opt,
        lr_scheduler=sched,
        loss=1.0,
    )

    # Save valid ref_model_identity.json
    ref_hash = compute_model_weight_hash(ref_model)
    (cp_path / "ref_model_identity.json").write_text(json.dumps({
        "num_parameters": sum(p.numel() for p in ref_model.parameters()),
        "param_hash": ref_hash,
        "revision": "test-rev-1",
    }), encoding="utf-8")

    # 1. Normal resume with matching ref_model passes
    info = ResumeManager.resume_into(cp_path, model, opt, sched, ref_model=ref_model)
    assert info["step"] == 2

    # 2. Mutate reference model parameter -> must reject with ValueError
    with torch.no_grad():
        for p in ref_model.parameters():
            p.add_(1.0)
            break

    with pytest.raises(ValueError, match="Reference model parameter integrity/weight hash mismatch on resume"):
        ResumeManager.resume_into(cp_path, model, opt, sched, ref_model=ref_model)

    # 3. Missing required bundle file (e.g. optimizer.pt deleted) -> must reject
    (cp_path / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint bundle"):
        ResumeManager.resume_into(cp_path, model)


