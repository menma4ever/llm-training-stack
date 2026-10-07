# Python API Reference — `llm-training-stack`

The `llm-training-stack` Python API provides a strictly typed, programmatic interface for configuring, preflighting, training, evaluating, and resuming LLM workloads.

---

## 📦 Module Hierarchy

```text
llm_training_stack/
├── config/              # Pydantic v2 schemas and YAML/JSON loaders
│   ├── schema.py        # TrainingJobConfig, ModelConfig, PeftConfig, etc.
│   └── loader.py        # ConfigLoader
├── preflight/           # Hardware detection, analytical model, empirical probe
│   ├── hardware.py      # HardwareInspector
│   ├── memory_model.py  # MemoryEstimator
│   ├── probe.py         # EmpiricalMemoryProbe
│   └── data_inspector.py# TokenizerInspector, DataInspector, PreflightInspectionSuite
├── pipelines/           # Execution workflows
│   ├── cpt.py           # CPTPipeline
│   ├── sft.py           # SFTPipeline (with prompt masking labels=-100)
│   ├── lora.py          # LoRAPipeline (PEFT adapters)
│   └── dpo.py           # DPOPipeline (Direct Preference Optimization)
├── checkpoints/         # Bundling and deterministic resume
│   ├── manager.py       # CheckpointManager
│   └── resume.py        # ResumeManager
├── eval/                # Evaluation & comparative analysis
│   ├── evaluator.py     # Evaluator (Loss & Perplexity)
│   └── comparator.py    # RunComparator
└── provenance/          # Environment fingerprinting & telemetry
    ├── manifest.py      # RunManifest
    └── event_logger.py  # StructuredEventLogger
```

---

## ⚙️ 1. Configuration (`llm_training_stack.config`)

All configuration structures inherit from Pydantic `BaseModel` with strict validation.

### `TrainingJobConfig`
Root configuration object representing an immutable job definition:

```python
from llm_training_stack.config.schema import (
    TrainingJobConfig, TaskType, ModelConfig, DatasetConfig,
    PeftConfig, OptimizerConfig, HardwareConfig, LoggingConfig
)

config = TrainingJobConfig(
    schema_version="1.0.0",
    task_type=TaskType.LORA,
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    ),
    dataset=DatasetConfig(
        dataset_name_or_path="synthetic",
        prompt_column="prompt",
        response_column="response",
        max_seq_length=1024,
    ),
    peft=PeftConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ),
    optimizer=OptimizerConfig(
        optimizer_type="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    ),
    hardware=HardwareConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
    ),
    logging=LoggingConfig(
        output_dir="./runs/my_experiment",
        save_steps=25,
        save_total_limit=3,
    ),
    epochs=1,
    max_steps=100,
    seed=42,
)
```

### `ConfigLoader`
Loads configuration from YAML or JSON files:

```python
from llm_training_stack.config.loader import ConfigLoader

config = ConfigLoader.load_from_file("examples/training_config.yaml")
```

---

## 🔬 2. Preflight & Hardware Inspection (`llm_training_stack.preflight`)

### `HardwareInspector`
Gathers comprehensive hardware metrics:

```python
from llm_training_stack.preflight.hardware import HardwareInspector

info = HardwareInspector.inspect()
# Returns: {
#   "platform": "...",
#   "python_version": "...",
#   "cpu_count_physical": 6,
#   "cpu_count_logical": 12,
#   "system_ram_total_gb": 32.0,
#   "cuda_available": False,
#   "recommended_device": "cpu"
# }
```

### `MemoryEstimator`
Calculates analytical static memory bounds:

```python
from llm_training_stack.preflight.memory_model import MemoryEstimator

estimate = MemoryEstimator.estimate(config, total_params=500_000_000)
# Returns: weights_mb, gradients_mb, optimizer_mb, activations_mb, estimated_total_gb
```

### `EmpiricalMemoryProbe`
Runs isolated 1-step micro-probes:

```python
from llm_training_stack.preflight.probe import EmpiricalMemoryProbe

# Run in an isolated subprocess to prevent host crash
report = EmpiricalMemoryProbe.run_probe(
    model="Qwen/Qwen2.5-0.5B",
    config=config,
    in_subprocess=True,
    timeout_seconds=60,
)
# Returns: probe_successful, verdict, measured_peak_allocated_mb, memory_headroom_percent
```

### `PreflightInspectionSuite`
Audits tokenizers and token distributions:

```python
from llm_training_stack.preflight.data_inspector import PreflightInspectionSuite

inspection = PreflightInspectionSuite.run_full_inspection(
    model_or_config=model,
    tokenizer=tokenizer,
    samples=["Sample prompt 1", "Sample prompt 2"],
    max_seq_length=512,
)
```

---

## 🚀 3. Pipelines (`llm_training_stack.pipelines`)

Workflows are executed through dedicated pipeline classes:

```python
from llm_training_stack.pipelines import (
    CPTPipeline,
    SFTPipeline,
    LoRAPipeline,
    DPOPipeline,
)

# Initialize pipeline
pipeline = LoRAPipeline(config)

# Run full training (or resume from existing bundle)
result = pipeline.train(resume_from=None)

print(result["final_loss"])
print(result["final_checkpoint"])
```

---

## 💾 4. Checkpoints & Resumption (`llm_training_stack.checkpoints`)

### `CheckpointManager`
Saves atomic bundles with state rotation:

```python
from llm_training_stack.checkpoints.manager import CheckpointManager

manager = CheckpointManager(output_dir="./runs/my_exp", save_total_limit=2)
saved_dir = manager.save_checkpoint(
    step=50,
    epoch=1.0,
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    data_index=200,
    loss=1.42,
)
```

### `ResumeManager`
Verifies integrity before loading:

```python
from llm_training_stack.checkpoints.resume import ResumeManager

audit = ResumeManager.inspect_checkpoint("./runs/my_exp/checkpoint-50")
assert audit["is_complete_bundle"] is True
```

---

## 📊 5. Provenance & Run Comparison (`llm_training_stack.provenance`, `eval`)

```python
from llm_training_stack.eval.comparator import RunComparator

comparison = RunComparator.compare_runs(["./runs/run_a", "./runs/run_b"])
print(comparison["markdown_report"])
```
