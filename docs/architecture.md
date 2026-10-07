# LLM Training Stack — Architecture & Technical Rationale

This document details the architectural design, component interactions, safety mechanisms, and research rationale behind the **LLM Training Stack**.

---

## 🏛️ System Architecture Overview

The LLM Training Stack is built on a modular, layered architecture designed for deterministic execution, strict resource bounds, and interface parity across Python API, CLI, and MCP.

```mermaid
graph TD
    subgraph Interfaces ["User & Orchestration Interfaces"]
        API["Python API (Typed SDK)"]
        CLI["Typer CLI (train-stack)"]
        MCP["Model Context Protocol (FastMCP)"]
    end

    subgraph ConfigPreflight ["Configuration & Validation Layer"]
        CFG["ConfigLoader & Pydantic Schemas<br/>(schema_version: 1.0.0)"]
        HW["HardwareInspector<br/>(CPU / RAM / CUDA / MPS)"]
        MEM["MemoryEstimator<br/>(Analytical 8B AdamW Model)"]
        PROBE["EmpiricalMemoryProbe<br/>(1-Step Subprocess Micro-Probe)"]
    end

    subgraph EnginePipelines ["Execution & Pipeline Engine"]
        CPT["CPTPipeline<br/>(Continued Pre-Training)"]
        SFT["SFTPipeline<br/>(Assistant Loss Masking labels=-100)"]
        LORA["LoRAPipeline<br/>(PEFT Low-Rank Adapters)"]
        DPO["DPOPipeline<br/>(Direct Preference Optimization)"]
        ACCEL["Hugging Face Accelerate<br/>(Device Placement / AMP / Distributed)"]
    end

    subgraph StateProvenance ["State, Provenance & Evaluation"]
        CKPT["CheckpointManager & ResumeManager<br/>(safetensors, optimizer, lr_sched, RNG, meta)"]
        PROV["RunManifest & StructuredEventLogger<br/>(Environment fingerprinting & events.jsonl)"]
        EVAL["Evaluator & RunComparator<br/>(Loss, Perplexity, Run Metrics)"]
    end

    API --> CFG
    CLI --> CFG
    MCP --> CFG

    CFG --> HW
    CFG --> MEM
    MEM --> PROBE
    PROBE --> EnginePipelines

    CPT --> ACCEL
    SFT --> ACCEL
    LORA --> ACCEL
    DPO --> ACCEL

    EnginePipelines --> CKPT
    EnginePipelines --> PROV
    EnginePipelines --> EVAL
```

---

## 🔬 Dual-Stage Memory Preflight Architecture

Memory estimation in LLM training often suffers from either overly optimistic static calculations or out-of-memory (OOM) crashes during mid-run execution. The stack solves this via a **dual-stage preflight architecture**:

```mermaid
flowchart TD
    A["User Submits TrainingJobConfig"] --> B["Stage 1: Analytical Memory Preflight"]
    
    subgraph Stage1 ["Stage 1: Analytical Estimation"]
        B --> C["Calculate Static Weights Memory<br/>Params × BytesPerParam"]
        B --> D["Calculate Gradients & Optimizer States<br/>AdamW: 8 bytes/param fp32 master"]
        B --> E["Calculate Activation Memory<br/>SeqLen × HiddenDim × Layers × AttentionHeads"]
        B --> F["Estimate Static Total Peak Memory"]
    end

    F --> G{"Does static estimate fit hardware RAM/VRAM?"}
    G -- No --> H["FAIL FAST: Raise ConfigValidationError<br/>Suggest lower batch size / seq_len / LoRA"]
    G -- Yes --> I["Stage 2: Empirical Micro-Probe"]

    subgraph Stage2 ["Stage 2: Empirical Subprocess Micro-Probe"]
        I --> J["Spawn Isolated Worker Subprocess"]
        J --> K["Load Micro-Model (or Target Weights)"]
        K --> L["Execute Single-Step Synthetic Forward Pass"]
        L --> M["Execute Single-Step Synthetic Backward Pass"]
        M --> N["Sample Peak Resident Set Size (RSS) / VRAM"]
        N --> O["Terminate Subprocess & Clean Up Memory"]
    end

    O --> P["Verify Measured RSS <= Available System RAM"]
    P -- Exceeded --> Q["ABORT: Measured allocation exceeds limits"]
    P -- Within Bounds --> R["APPROVE: Proceed to full pipeline execution"]
```

### Preflight Rationale:
- **Never Promise Unmeasured Fit**: Analytical approximations are tagged `[ESTIMATED]` because dynamic allocation, fragmentation, CUDA kernel workspaces, and PyTorch caching cannot be predicted purely from parameter counts.
- **Subprocess Isolation**: The empirical micro-probe executes in an isolated Python subprocess so that any potential OOM or memory fragmentation does not contaminate the primary orchestrator session.

---

## 🔄 Deterministic Resumption Lifecycle

The stack provides mathematically exact training resumption: an interrupted and resumed run produces identical loss trajectories to an uninterrupted baseline run.

```mermaid
sequenceDiagram
    autonumber
    actor Orchestrator as Orchestrator / CLI
    participant Engine as Pipeline Engine
    participant Checkpointer as CheckpointManager
    participant Storage as Disk Storage

    Orchestrator->>Engine: train(resume_from=None)
    loop Each Step
        Engine->>Engine: Forward & Backward & Step
        opt Every save_steps
            Engine->>Checkpointer: save_checkpoint(step, epoch, model, opt, sched)
            Checkpointer->>Storage: model.safetensors (Weights)
            Checkpointer->>Storage: optimizer.pt (AdamW moments)
            Checkpointer->>Storage: lr_scheduler.pt (Schedule state)
            Checkpointer->>Storage: rng_state.pt (CPU, CUDA, NumPy, Random)
            Checkpointer->>Storage: metadata.json (step, epoch, data_index, loss)
            Checkpointer->>Checkpointer: Rotate checkpoints (keep save_total_limit)
        end
    end

    Note over Engine,Storage: Interruption Occurs (Crash, Preemption, or Stop)

    Orchestrator->>Engine: train(resume_from="checkpoint-N")
    Engine->>Checkpointer: inspect_and_verify("checkpoint-N")
    Checkpointer-->>Engine: Complete bundle verified
    Engine->>Engine: Load safetensors weights into model
    Engine->>Engine: Load optimizer.pt state into AdamW
    Engine->>Engine: Load lr_scheduler.pt into scheduler
    Engine->>Engine: Restore torch.set_rng_state, numpy, python seeds
    Engine->>Engine: Fast-forward dataloader to metadata.data_index
    Engine->>Engine: Resume step N+1 (exact loss continuity achieved)
```

---

## 🛡️ Model Context Protocol (MCP) Security Architecture

When an autonomous AI agent interacts with training infrastructure, execution boundaries must prevent unauthorized compute expenditures, credential theft, and arbitrary code execution.

```mermaid
flowchart TD
    subgraph AgentClient ["AI Agent / LLM Client"]
        Agent["Autonomous Agent"]
    end

    subgraph MCPServer ["Safe MCP Server (FastMCP)"]
        Gate["MCP Policy Validator"]
        
        subgraph ReadTools ["Read & Telemetry Tools (Safe)"]
            T1["inspect_hardware()"]
            T2["estimate_memory()"]
            T3["inspect_checkpoint()"]
            T4["compare_runs()"]
        end
        
        subgraph LaunchTools ["Protected Launch Tool (Guarded)"]
            L1["launch_training()"]
        end
    end

    subgraph OSCompute ["System Host & Compute Hardware"]
        SysExec["PyTorch Training Execution"]
    end

    Agent -->|"call: inspect_hardware / estimate_memory"| ReadTools
    ReadTools -->|"Returns JSON Telemetry"| Agent

    Agent -->|"call: launch_training(config, authorized=False)"| Gate
    Gate -->|"authorized is False"| Reject1["DENIED: EXECUTION_DENIED<br/>Requires explicit authorized=True"]
    Reject1 --> Agent

    Agent -->|"call: launch_training(config, authorized=True)"| Gate
    Gate -->|"Check Env: TRAIN_STACK_ALLOW_LAUNCH"| CheckEnv{"TRAIN_STACK_ALLOW_LAUNCH == 1?"}
    CheckEnv -- No --> Reject2["DENIED: EXECUTION_DENIED_BY_OPERATOR_POLICY<br/>Server operator has not enabled execution"]
    CheckEnv -- Yes --> CheckCode{"config.model.trust_remote_code?"}
    CheckCode -- True --> CheckRC{"TRAIN_STACK_ALLOW_REMOTE_CODE == 1?"}
    CheckRC -- No --> Reject3["DENIED: REMOTE_CODE_POLICY_VIOLATION<br/>Arbitrary remote code forbidden by default"]
    CheckRC -- Yes --> Run[Allow Execution]
    CheckCode -- False --> Run

    Run --> SysExec
```

### Security Invariants:
1. **No Arbitrary Execution**: There is **no shell tool**, **no `eval()` tool**, and no arbitrary file system write capability exposed over MCP.
2. **Double Authorization Barrier**:
   - The AI agent must supply `authorized=True`.
   - The host system operator must explicitly export `TRAIN_STACK_ALLOW_LAUNCH=1`.
3. **Remote Code Gating**: Models with `trust_remote_code=True` are rejected unless the host administrator explicitly set `TRAIN_STACK_ALLOW_REMOTE_CODE=1`.

---

## 🎯 Training Workflows & Rationale

### 1. Continued Pre-Training (CPT)
- **Goal**: Ingest domain-specific corpora (legal, medical, code) to adapt base vocabulary distributions.
- **Loss Function**: Standard autoregressive cross-entropy over causal tokens.
- **Dataset Packing**: Supports concatenation of token sequences with end-of-sequence delimiters up to `max_seq_length`.

### 2. Supervised Fine-Tuning (SFT) with Masking
- **Assistant-Only Loss Masking**: Crucial for conversational alignment. Prompt tokens are assigned `labels = -100` so that gradients are only computed with respect to assistant response tokens.
- **Empty Prompt / Target Validation**: Rejects malformed examples with empty prompts or empty targets during dataset preprocessing.

### 3. Parameter-Efficient Fine-Tuning (LoRA / PEFT)
- **Architecture**: Injects trainable rank-decomposition matrices ($W + \Delta W$, where $\Delta W = B \times A$) into attention projection layers (`q_proj`, `v_proj`).
- **Memory Efficiency**: Reduces trainable parameters to $< 1\%$ of base model size, lowering optimizer memory from $8 \times N$ bytes to $8 \times M$ bytes ($M \ll N$).

### 4. Direct Preference Optimization (DPO)
- **Architecture**: Optimizes policies directly on preference pairs $(x, y_w, y_l)$ without needing an explicit separate reward model:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
- **Input Validation**: Automatically rejects identical pairs ($y_w = y_l$) or empty responses before starting training.
