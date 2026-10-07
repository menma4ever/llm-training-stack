# Reproducibility & Deterministic Resumption Guide

Achieving true scientific and operational reproducibility in deep learning requires more than fixing a random seed. In distributed environments and long-running training jobs, runs may experience interruptions due to spot instance preemption, cluster maintenance, or power failures.

This document details how `llm-training-stack` guarantees **deterministic resumption** and **full run provenance**.

---

## 🎯 The Determinism Guarantee

When a training job is interrupted at step $N$ and resumed from step $N$:
$$\mathcal{L}_{\text{resumed}}(N+1) = \mathcal{L}_{\text{uninterrupted}}(N+1) \pm 0.00000$$

The loss trajectory of a resumed run matches the trajectory of an uninterrupted run within floating-point tolerance. This guarantee is continuously verified by `tests/test_resume_continuity.py`.

---

## 📦 Anatomy of a Complete Checkpoint Bundle

Saving only model weights (`model.safetensors` or `pytorch_model.bin`) is fundamentally insufficient for deterministic resumption. A standard weight-only checkpoint causes immediate divergence for three reasons:
1. **Cold Optimizer States**: AdamW maintains running first moments ($m_t$) and second moments ($v_t$). Initializing an optimizer from zero resets adaptive learning rates, causing an immediate loss spike.
2. **Desynchronized LR Schedules**: Cosine decay or linear warmup schedulers reset to step 0 if not persisted.
3. **Divergent RNG Streams**: Dropout masks and stochastic token masking rely on pseudo-random generator state.

`CheckpointManager` guarantees that every checkpoint folder is an atomic, self-contained bundle:

```text
runs/lora_experiment/checkpoint-100/
├── model.safetensors       # Model weights in secure, zero-copy format
├── optimizer.pt            # Full AdamW optimizer state dictionary
├── lr_scheduler.pt         # Learning rate scheduler state dictionary
├── rng_state.pt            # Complete RNG bundle (PyTorch CPU, CUDA, NumPy, Random)
└── metadata.json           # Step, epoch, sample index, loss, loss history
```

### Complete State Capture in `rng_state.pt`:
```python
rng_bundle = {
    "torch_cpu": torch.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    "numpy": np.random.get_state(),
    "python": random.getstate(),
}
```

---

## 🛡️ Preflight Audit: `ResumeManager`

Before resuming any training job, `ResumeManager.inspect_checkpoint()` validates bundle integrity:
- Checks for file existence of all 5 bundle components.
- Verifies that `model.safetensors` can be loaded.
- Checks that `step` and `epoch` in `metadata.json` are positive integers/floats.
- Checks that the optimizer state matches the architecture parameter keys.

If any bundle component is missing or corrupt, resumption halts immediately with an actionable error rather than silently resuming with uninitialized optimizer state.

---

## 📜 Run Manifest & Telemetry Provenance

Every run directory automatically generates two immutable provenance artifacts:

### 1. `manifest.json` (Environment Fingerprint)
Captures the complete execution context:
- **Environment**:
  - Python version & platform architecture
  - PyTorch version (`torch.__version__`)
  - Transformers, PEFT, TRL, Accelerate library versions
  - Git commit hash & working tree clean/dirty status
- **Hardware Specs**:
  - CPU model, physical/logical cores, system RAM
  - CUDA availability, device count, device names, VRAM capacity
- **Immutable Job Configuration**:
  - Full serialized `TrainingJobConfig` JSON.

### 2. `events.jsonl` (Append-Only Event Stream)
Every training lifecycle event (initialization, step progress, checkpoint saved, evaluation metric, completion) is logged as an append-only JSON line with timestamp, step, epoch, loss, learning rate, and memory consumption.

---

## 🔬 How to Verify Locally

Run the automated resumption test:

```bash
pytest tests/test_resume_continuity.py -v
```

This test:
1. Trains a baseline model for 4 steps and records $[\mathcal{L}_1, \mathcal{L}_2, \mathcal{L}_3, \mathcal{L}_4]$.
2. Trains a second model for 2 steps, checkpoints it to disk, terminates execution.
3. Re-instantiates the model, loads the checkpoint bundle, fast-forwards the dataset, and trains for 2 remaining steps.
4. Asserts that the resumed losses $[\mathcal{L}_3, \mathcal{L}_4]$ match the baseline losses within machine precision.

---

## 🔬 Evidence Partitioning & Boundary Distinctions

In accordance with strict empirical reporting (CEO Reviews 21 & 22):

### 1. Source-Host Native Multi-Process Resume vs Installed Virtualenv Evidence
- **Source-Host Multi-Process Resume**: Documented in `artifacts/NATIVE_UPSTREAM_RESUME_EVIDENCE.log` (3,392 B, SHA256 `bcb0f6bbf9d839ac4355032c8ebd73209e053a5a34152d1ab61c03f9b5ab3e7c`). Demonstrates native CPT/SFT resume continuity across 6 distinct OS subprocesses with bitwise `0.000000e+00` maximum parameter diff and matching loss continuation.
- **Installed Candidate Virtualenv Evidence**: Documented in `artifacts/FINAL_INSTALLED_CANDIDATE_EVIDENCE.log` (15,408 B, SHA256 `8b600ff928767c30f0724e08be1326510475dde6efa21c3ca1ab95700ec7982c`). Proves pip check exit 0, site-packages module provenance, CLI entrypoint invocation, lifecycle unack/ack cancellation, and MCP tools under clean installed candidate environment.

### 2. Model Revision vs Source Git Commit
- **Model Revision**: The pinned open-weight model target is `HuggingFaceTB/SmolLM-135M` pinned to Hugging Face commit revision `1d461723eec654e65efdc40cf49301c89c0c92f4`.
- **Source Code Commit**: The training run was launched from local repository Git commit `5d5811746bda06e6bc7662fcb46b8d21ce2a5650` with `is_dirty: true` (as faithfully recorded in `artifacts/installed_candidate_run/manifest.json`).

### 3. Pinned Fixture Scope vs Broad Model Quality
- The 4-step SmolLM-135M CPU adaptation run (`artifacts/installed_candidate_run/`) operates on 10 training samples and evaluates on 5 held-out validation samples (`max_seq_length=64`).
- It demonstrates empirical loss reduction (`4.3873 -> 3.9773`, -9.34%) and perplexity reduction (`80.423 -> 53.373`, -33.63%) with `is_comparable: true` (invoked with `strict_compatibility=False`).
- **Scope Limitation**: This verifies CPU tensor arithmetic, loss computation, checkpointing, and evaluation data flow on a bounded fixture. It does NOT assert general model capability or benchmark quality.

### 4. Portable Unique-Output Scripts & Historical Retention
- Verification harnesses and execution scripts must always specify unique, non-colliding output directories (e.g. `--output-dir ./runs/run_<timestamp>`).
- All prior historical runs (`artifacts/installed_candidate_run/`, `artifacts/real_smoke_run/`, `artifacts/smoke_run/`) are strictly retained intact to prevent regression and ensure permanent auditability.

### 5. Standalone Reproduction Discipline (CEO Delta 23 Mandate)
- **Standalone Sanitized Checkout**: Reproduction commands must execute cleanly from an isolated distribution checkout without assuming fixed parent paths or ambient repository state.
- **Explicit Inputs & Unique Outputs**: Every command requires explicit inputs (`--dataset ./data/...`) and unique output paths (`--output-dir ./runs/repro_<timestamp>`) to prevent mutating existing accepted run artifacts.
- **Clean Constrained Environment**: Reproduction environments must be provisioned using exact frozen constraints (`pip install -r constraints.txt`), guaranteeing identical library builds without unconstrained float.
