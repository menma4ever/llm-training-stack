# Single-GPU CUDA Validation Protocol & Execution Plan

This document establishes the empirical validation protocol for executing `llm-training-stack` workloads on physical NVIDIA CUDA GPU hardware (Phase 2 milestone).

> [!IMPORTANT]
> **Status**: Planned Empirical Protocol (Hardware-Unverified on Host).
> This protocol specifies the exact execution commands, instrumentation hooks, and acceptance criteria required for hardware sign-off once physical GPU compute is provisioned. **No benchmarks or GPU measurements are claimed without physical run telemetry.**

---

## 1. Validation Objectives

The single-GPU validation milestone evaluates four core architectural guarantees:

1. **Functional Training Execution**: Successful end-to-end execution of Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), LoRA adapter training, and Direct Preference Optimization (DPO) on CUDA devices.
2. **Memory Profiling & AMP Efficiency**: Verification of Automatic Mixed Precision (`fp16` and `bf16`), measurement of static vs. dynamic activation memory via `torch.cuda.max_memory_allocated()`, and empirical probe accuracy.
3. **Throughput & Step Latency**: Measurement of training throughput (samples/sec and tokens/sec) under steady-state conditions, excluding JIT warmup and initial model weight allocation.
4. **Deterministic Checkpointing & Resume**: Bitwise and metric-level continuity of loss trajectories when resuming an interrupted run from a complete checkpoint bundle.

---

## 2. Test Fixture & Model Matrix

To balance verification speed and realistic model dynamics, validation will be conducted across two reference tiers:

| Tier | Target Architecture | Parameter Count | Precision Modes | Dataset Target | Primary Validation Purpose |
|---|---|---|---|---|---|
| **Micro-Tier (Smoke)** | `HuggingFaceTB/SmolLM-135M` | 135M | `fp16`, `bf16`, `fp32` | `data/real_sft_dataset.jsonl` (10 samples, max seq len 64) | Fast CI/CD sanity check, gradient flow, device placement |
| **Production-Tier** | `Qwen/Qwen2.5-0.5B` / `meta-llama/Llama-3.2-1B` | 500M–1B | `bf16` (Ampere+), `fp16` | Sanitized conversation corpus (1,000 samples, seq len 512/1024) | Realistic VRAM scaling, throughput, optimizer state sharding |

---

## 3. Step-by-Step Validation Protocol

### Step 3.1: Hardware & Device Inspection
Verify that PyTorch identifies the CUDA runtime, device capability, driver version, and VRAM budget:

```bash
# Verify CLI device discovery
train-stack inspect
```

**Acceptance Criteria**:
- Output detects `Device: cuda:0`.
- Reports accurate GPU model name (e.g., `NVIDIA GeForce RTX 4090` or `NVIDIA A100-SXM4-80GB`).
- VRAM total capacity and driver version match `nvidia-smi`.

---

### Step 3.2: Analytical Preflight & Empirical Memory Probe
Execute the two-stage preflight validation on GPU:

```bash
train-stack preflight --config examples/training_config.yaml
```

**Acceptance Criteria**:
- Stage 1 (Analytical): Computes static parameter memory, AdamW 8-byte state estimate, and activation upper bound without allocating GPU memory.
- Stage 2 (Empirical Micro-Probe): Spawns an isolated 1-step subprocess on `cuda:0`, measures actual `peak_vram_bytes`, asserts peak memory is strictly within configured RAM budget, and cleanly releases VRAM upon termination without CUDA context leaks.

---

### Step 3.3: Real Supervised Fine-Tuning (SFT) Run
Launch a bounded 50-step training job with LoRA adapters:

```bash
python -m llm_training_stack.cli.main train \
  --config examples/training_config.yaml \
  --output-dir ./runs/cuda_validation_sft \
  --device "cuda:0" \
  --precision "bf16"
```

**Key Measurements to Log**:
1. **Initial VRAM (Post-Model Load)**: Base model weights in VRAM.
2. **Peak VRAM During Forward/Backward**: Peak allocation recorded via `torch.cuda.max_memory_allocated()`.
3. **Steady-State Step Duration**: Milliseconds per optimizer step across steps 10–50.
4. **Effective Throughput**:
   $$\text{Throughput} = \frac{\text{Batch Size} \times \text{Gradient Accumulation} \times \text{Sequence Length}}{\Delta t_{\text{step}} \text{ (seconds)}} \quad \text{[tokens/sec]}$$

---

### Step 3.4: Deterministic Checkpoint & Resume Validation
Verify that saving at step $k$ and resuming from step $k$ preserves loss trajectory and optimizer state:

```bash
# 1. Run training for 20 steps, saving checkpoint at step 10
python -m llm_training_stack.cli.main train \
  --config examples/training_config.yaml \
  --output-dir ./runs/cuda_checkpoint_baseline \
  --max-steps 20 \
  --save-steps 10

# 2. Audit checkpoint bundle completeness
train-stack resume --checkpoint ./runs/cuda_checkpoint_baseline/checkpoint-10

# 3. Resume from checkpoint-10 for an additional 10 steps (up to step 20)
python -m llm_training_stack.cli.main train \
  --config examples/training_config.yaml \
  --output-dir ./runs/cuda_checkpoint_resumed \
  --resume ./runs/cuda_checkpoint_baseline/checkpoint-10 \
  --max-steps 20
```

**Acceptance Criteria**:
- `checkpoint-10` bundle contains: `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pt`, `manifest.json`.
- Step 11 loss in resumed run matches step 11 loss in baseline run within floating-point tolerance ($\Delta \text{loss} \le 10^{-4}$).
- Total trained step count equals 20.

---

### Step 3.5: Run Comparison & Artifact Audit
Generate an analytical Markdown comparison across baseline and resumed executions:

```bash
train-stack compare ./runs/cuda_checkpoint_baseline ./runs/cuda_checkpoint_resumed
```

**Acceptance Criteria**:
- Output confirms `is_comparable: true`.
- Verifies final loss parity, total duration telemetry, and memory profile.
- Emits structured event logs to `./runs/events.jsonl` with valid SHA256 checksums.

---

## 4. Hardware Verification Sign-Off Template

When physical CUDA runs are executed, results will be permanently recorded in this matrix:

```markdown
### CUDA Validation Sign-Off
- **Date**: YYYY-MM-DD
- **GPU Model**: [e.g., NVIDIA RTX 4090 24GB / A100 80GB]
- **CUDA Toolkit**: [e.g., 12.4] | **Driver**: [e.g., 550.54.14]
- **PyTorch Build**: [e.g., 2.4.0+cu124]
- **Target Model**: SmolLM-135M / Qwen2.5-0.5B
- **Precision**: BF16 (AMP)
- **Peak Measured VRAM**: X.XX GB
- **Throughput**: XXXX tokens/sec
- **Checkpoint Resume Parity**: Verified (Delta < 1e-4)
- **Evidence Log Path**: artifacts/cuda_validation_evidence.log
```
