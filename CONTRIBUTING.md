# Contributing to LLM Training Stack

Thank you for your interest in contributing to the **LLM Training Stack**! We welcome bug reports, architectural critiques, documentation improvements, and pull requests.

---

## 📐 Development Guidelines

1. **Python Standards**:
   - Python 3.10+ compatible.
   - Strict typing with type hints throughout all public APIs.
   - Pydantic v2 schemas for all configuration models.
2. **Deterministic Resumption**:
   - Any new optimizer, scheduler, or dataloader logic must preserve state restoration guarantees.
   - Checkpoints must bundle weights (`safetensors`), `optimizer.pt`, `scheduler.pt`, `rng_state.pt`, and `metadata.json`.
3. **Honest Support Claims**:
   - Do not claim GPU, multi-GPU, FSDP2, or DeepSpeed fit without empirical evidence.
   - All preflight memory estimates must distinguish analytical formulas from empirical measurements.
4. **Secret & Privacy Hygiene**:
   - Never commit API keys, GitHub tokens, Google auth credentials, or private local user directory paths (e.g., `/home/username/...` or relative placeholders).
   - Scrub personal emails from code, docstrings, and configuration files.

---

## 🧪 Testing Requirements

Before opening a pull request, run the test suite and ensure all tests pass:

```bash
pytest tests/ -v
```

Include new unit and integration tests for any added features or bug fixes.
