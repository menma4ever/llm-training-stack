# Hardware & Framework Compatibility Matrix

This document provides a truthful, evidence-grounded audit of tested hardware platforms, distributed training backends, precision formats, and dependency specifications for `llm-training-stack`.

---

## 🛡️ Truthful Verification Charter

In accordance with strict empirical engineering principles:
1. **Never Promise Unmeasured Fit**: Analytical memory approximations are explicitly marked as `[ESTIMATED]`. Real memory requirements are only confirmed via empirical execution (`[MEASURED]`).
2. **Explicit Hardware Verification Gating**: Features integrated into the codebase that have not been executed on physical hardware are marked as **Hardware-Unverified (Experimental)**.
3. **Bounded OS Scope & CI Distinction**: Only **Windows 11 x86_64 CPU** has been empirically verified in the active host environment. The checked-in GitHub Actions CI workflow (`.github/workflows/ci.yml`) defines a matrix across Ubuntu, macOS, and Windows, but this is a checked-in remote CI definition, NOT a passed multi-OS run on host. Linux, macOS, and physical GPU backends remain **Empirically Unverified on Host** until verified on real infrastructure.
4. **No Fabricated Benchmarks**: Multi-GPU, FSDP2, and DeepSpeed capabilities are architecturally integrated and unit-tested via mocking/stubs, but physical multinode runs remain on the roadmap until hardware evidence is collected on real clusters.

---

## 📊 Compute & Hardware Support Matrix

| Compute Platform | Support Status | Verification Level | Backend / Runtime | Host Telemetry & Verification Evidence |
|---|---|---|---|---|
| **Local CPU (Windows x86_64)** | 🟢 **Verified & Tested** | Tier 1 (Host Verified) | PyTorch CPU / OneDNN | Full test suite passed (114 automated tests across 12 test modules in `TEST_SUITE_EVIDENCE_14.log`). Verified on Intel Core i5-12400F (6C/12T), 32 GB RAM. CPT, SFT, LoRA, and DPO training loops executed end-to-end. |
| **Local CPU (Linux / macOS)** | ⚠️ **Architecturally Supported** | Empirically Unverified on Host | PyTorch CPU | Cross-platform abstractions in place; empirical host execution pending Linux/macOS runners. Checked-in CI workflow is not a passed host run. |
| **Single GPU** (NVIDIA CUDA) | ⚠️ **Architecturally Supported** | Hardware-Unverified on Host | PyTorch CUDA / SDPA / AMP | Code paths, device placement (`cuda:0`), mixed precision (`fp16`/`bf16`), and memory tracking fully implemented. Marked unverified until physical CUDA GPU telemetry is attached. |
| **Multi-GPU (DDP)** | ⚠️ **Architecturally Supported** | Hardware-Unverified on Host | `torch.distributed` / Accelerate | Standard DistributedDataParallel process spawning and gradient synchronization integrated via Accelerate. Marked experimental until multi-device verification. |
| **Multi-GPU (FSDP2)** | ⚠️ **Architecturally Supported** | Hardware-Unverified on Host | PyTorch FSDP2 (`torch.distributed.fsdp`) | Per-layer sharded parameter and optimizer state hooks configured. Marked experimental until physical multi-GPU cluster validation. |
| **DeepSpeed ZeRO (1/2/3)** | ⚠️ **Architecturally Supported** | Hardware-Unverified on Host | DeepSpeed / Accelerate plugin | Configuration templates and engine hooks integrated into job schemas. Hardware execution pending multi-GPU server availability. |
| **Apple Silicon (MPS)** | ⚠️ **Architecturally Supported** | Hardware-Unverified on Host | PyTorch MPS | Fallback device mapping available in `HardwareInspector`. Untested on physical Apple hardware. |
| **Multi-Node Distributed** | 🗺️ **Roadmap Extension** | Interface Design Only | Torchrun rendezvous / Slurm / MPI | Extension architecture specified in documentation. No unmeasured claims of multi-node execution are made. |

---

## 🔬 Host Verification Audit (Local Environment)

The following host specifications represent the active testing baseline where all automated test suites, CLI workflows, and candidate smoke runs pass:

- **Host Processor**: Intel® Core™ i5-12400F (6 physical cores, 12 logical threads, 2.50 GHz base, 4.40 GHz turbo)
- **Host System RAM**: 32.0 GB DDR4
- **Operating System**: Windows 11 Pro 64-bit (Build 10.0.26200)
- **Active Python**: 3.11.15 (64-bit)
- **PyTorch Build**: PyTorch 2.14.1+cpu
- **Local CUDA Hardware**: None detected (`torch.cuda.is_available() == False`)
- **Execution Mode**: Local CPU execution with PyTorch SDPA attention and PyTorch AdamW

---

## ⚙️ Precision & Numerical Formats

| Format | CPU Status | CUDA Status | Memory Footprint (Weights) | Notes |
|---|---|---|---|---|
| **FP32** (`float32`) | 🟢 **Verified** | 🟢 Supported | 4.0 bytes / parameter | Standard baseline for CPU testing and unit test validation. |
| **BF16** (`bfloat16`) | 🟢 Supported (Native/Emulated) | ⚠️ Unverified (Requires Ampere+) | 2.0 bytes / parameter | Supported natively by PyTorch on compatible CPUs; recommended for Ampere/Ada/Hopper GPUs. |
| **FP16** (`float16`) | ❌ **Enforced Rejection (`UNSUPPORTED_CONFIGURATION`)** | ⚠️ Unverified (Requires GPU) | 2.0 bytes / parameter | Strictly rejected on CPU execution by `EmpiricalMemoryProbe` and autograd; requires physical CUDA GPU with `GradScaler`. |
| **8-bit (BitsAndBytes)** | ❌ Unsupported on CPU | ⚠️ Hardware Dependent | 1.0 byte / parameter | Requires `bitsandbytes` CUDA binaries. |
| **4-bit (QLoRA / NF4)** | ❌ Unsupported on CPU | ⚠️ Hardware Dependent | ~0.55 bytes / parameter | Requires `bitsandbytes` with double quantization. |

---

## 📦 Python & Dependency Compatibility

The following table reflects the exact tested dependency constraints verified in the installed candidate virtual environment (CPython 3.11 candidate venv, `constraints.txt`):

| Dependency | Minimum Declared | Exact Tested Constraint (Candidate venv) | Purpose | Group |
|---|---|---|---|---|
| **Python** | `>= 3.10` | `3.11.15` | Language runtime | Core |
| **torch** | `>= 2.0.0` | `2.14.1+cpu` | Core tensor & autograd engine | Core |
| **transformers** | `>= 4.40.0` | `5.19.0` | Hugging Face model architectures | Core |
| **accelerate** | `>= 0.28.0` | `1.15.0` | Device placement & training orchestration | Core |
| **peft** | `>= 0.10.0` | `0.21.2` | LoRA / Parameter-Efficient Fine-Tuning | Core |
| **trl** | `>= 0.8.0` | `1.14.1` | SFT & DPO training utilities | Core |
| **datasets** | `>= 2.18.0` | `5.1.0` | Data loading, tokenization, batching | Core |
| **safetensors** | `>= 0.4.0` | `0.8.0` | Secure, zero-copy weight checkpointing | Core |
| **pydantic** | `>= 2.5.0` | `2.13.5` | Configuration schema validation | Core |
| **typer** | `>= 0.9.0` | `0.27.3` | CLI command line interface | Core |
| **rich** | `>= 13.0.0` | `15.0.0` | Terminal formatting and tables | Core |
| **pyyaml** | `>= 6.0` | `6.0.3` | Configuration file parsing | Core |
| **psutil** | `>= 5.9.0` | `7.2.2` | Process tree management & memory RSS tracking | Core |
| **mcp** | `>= 1.0.0` | `1.30.0` | Model Context Protocol server | Optional (`[mcp]`) |
| **pytest** | `>= 8.0.0` | `8.3.5` | Automated test suite execution | Optional (`[test]`, `[dev]`) |
| **pytest-cov** | `>= 4.1.0` | `6.0.0` | Code coverage reporting | Optional (`[test]`, `[dev]`) |
| **ruff** | `>= 0.3.0` | `0.9.9` | Code linting and style enforcement | Optional (`[dev]`) |
| **black** | `>= 24.0.0` | `24.10.0` | Code formatting | Optional (`[dev]`) |
| **mypy** | `>= 1.9.0` | `1.15.0` | Static type checking | Optional (`[dev]`) |

---

## 🗺️ Distributed Multi-Node Roadmap

The path from single-host execution to distributed multi-node clusters proceeds in three strictly audited phases:

```
[Phase 1: Local Verified Baseline] (CURRENT)
  ├── Host CPU verified with PyTorch 2.14.1+cpu (Windows 11)
  ├── Memory preflight (analytical + 1-step empirical probe)
  └── Full bundle deterministic checkpointing & resume
         │
         ▼
[Phase 2: Single & Multi-GPU Validation] (NEXT CLUSTER RELEASE)
  ├── Single-GPU CUDA verification with SDPA and FP16/BF16 AMP
  ├── Multi-GPU DDP verification with NCCL backend
  ├── FSDP2 sharded parameter & optimizer state verification
  └── DeepSpeed ZeRO-1/2/3 integration verification
         │
         ▼
[Phase 3: Multi-Node Distributed Cluster] (ROADMAP EXTENSION)
  ├── Torchrun rendezvous (c10d backend across nodes)
  ├── Slurm batch scheduler launcher integration
  ├── Elastic training node recovery and checkpoint migration
  └── Distributed checkpoint sharding across high-speed parallel filesystems (NFS/Lustre)
```

No claims of Phase 2 or Phase 3 production-readiness will be published until empirical execution logs and hardware benchmarks are collected on verified GPU infrastructure.
