# CLI Reference — `train-stack`

`train-stack` is the official command-line interface for the LLM Training Stack, built with [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich).

---

## 📋 Command Overview

| Command | Description | Core Flags / Arguments |
|---|---|---|
| `train-stack inspect` | Inspect local host hardware, CPU, RAM, CUDA availability | None |
| `train-stack preflight` | Execute analytical memory estimation and empirical micro-probe | `--config, -c` |
| `train-stack train` | Launch synchronous training run for CPT, SFT, LoRA, or DPO | `--config, -c`, `--resume` |
| `train-stack submit` | Submit managed asynchronous training job to lifecycle manager | `--config, -c`, `--resume` |
| `train-stack status` | Query execution status of an active or finished training job | `[run_dir]` |
| `train-stack logs` | Tail structured telemetry events and step losses | `[run_dir]`, `--tail, -n` |
| `train-stack cancel` | Signal cooperative cancellation to a running training job | `[run_dir]` |
| `train-stack eval` | Evaluate causal LM cross-entropy loss and perplexity on test split | `--model, -m`, `--dataset, -d`, `-o` |
| `train-stack resume` | Audit and verify checkpoint bundle completeness & integrity | `--checkpoint` |
| `train-stack compare` | Compare multiple training runs across metrics & compatibility | `[run_dirs...]` |
| `train-stack mcp` | Launch the safe Model Context Protocol (MCP) server | None |

---

## 🔍 `train-stack inspect`

Audits local host architecture, memory limits, and hardware accelerators.

```bash
train-stack inspect
```

---

## 🔬 `train-stack preflight`

Runs analytical static memory calculation and an empirical 1-step forward/backward probe inside an isolated subprocess with strict memory ceiling and timeout limits.

```bash
train-stack preflight --config examples/training_config.yaml
```

---

## 🚀 `train-stack train`

Launches a synchronous training pipeline (CPT, SFT, LoRA, or DPO) based on the specified configuration.

```bash
# Standard training launch
train-stack train --config examples/training_config.yaml

# Resuming from a previously saved checkpoint bundle
train-stack train --config examples/training_config.yaml --resume ./runs/lora_demo/checkpoint-5
```

---

## ⚡ `train-stack submit`

Submits an asynchronous training job to the background lifecycle manager. Emits a unique `job_id` and initial state.

```bash
train-stack submit --config examples/training_config.yaml
```

---

## 📊 `train-stack status`

Queries the live status of an asynchronous job by inspecting `job_state.json`.

```bash
train-stack status ./runs/job_1234
```

---

## 📜 `train-stack logs`

Tails structured JSONL telemetry events for step losses, throughput (tokens/sec), and durations.

```bash
train-stack logs ./runs/job_1234 --tail 50
```

---

## 🛑 `train-stack cancel`

Requests cooperative cancellation of an active training job.

```bash
train-stack cancel ./runs/job_1234
```

---

## 📈 `train-stack eval`

Computes held-out cross-entropy loss and perplexity with response-only loss masking (`labels = -100`) and deterministic dataset fingerprinting.

```bash
train-stack eval --model ./runs/lora_demo/checkpoint-10 --dataset ./data/eval_data.jsonl -o ./runs/eval_report.json
```

---

## 💾 `train-stack resume`

Audits a saved checkpoint directory to verify that it forms a complete, uncorrupted bundle capable of deterministic resumption (`safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pt`, `metadata.json`).

```bash
train-stack resume --checkpoint ./runs/lora_demo/checkpoint-10
```

---

## ⚖️ `train-stack compare`

Parses `manifest.json` files and held-out evaluation reports across two or more runs, performing strict compatibility validation on dataset fingerprints and tokenizers.

```bash
train-stack compare ./runs/run_a ./runs/run_b
```

---

## 🤖 `train-stack mcp`

Starts the Model Context Protocol (MCP) server over standard I/O for autonomous AI agent integration.

```bash
train-stack mcp
```

### Environment Variables & Security Policies:
- `TRAIN_STACK_ALLOW_LAUNCH=1`: Operator flag required to authorize training launches over MCP.
- `TRAIN_STACK_ALLOW_PROBE=1`: Operator flag required to authorize empirical memory probes over MCP.
- `TRAIN_STACK_ALLOW_REMOTE_CODE=1`: Operator flag required to authorize `trust_remote_code=True`.
- `TRAIN_STACK_ALLOWED_ROOTS`: Semicolon-delimited allowed directory roots for filesystem access.
