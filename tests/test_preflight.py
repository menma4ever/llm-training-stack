"""Tests for hardware inspection, memory estimation, data & tokenizer auditing, empirical probing,
resource identity classification, and preflight policy contract.
"""

import copy
from pathlib import Path
import psutil
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_training_stack.config.schema import (
    TrainingJobConfig,
    TaskType,
    ModelConfig,
    DatasetConfig,
    HardwareConfig,
    OptimizerConfig,
)
from llm_training_stack.preflight.hardware import HardwareInspector
from llm_training_stack.preflight.memory_model import MemoryEstimator
from llm_training_stack.preflight.probe import EmpiricalMemoryProbe, _get_aggregate_rss_mb
from llm_training_stack.preflight.data_inspector import (
    TokenizerInspector,
    DataInspector,
    PreflightInspectionSuite,
)
from llm_training_stack.preflight.resource_identity import (
    ResourceClassifier,
    ResourceIdentity,
    ResourceKind,
)
from llm_training_stack.preflight.policy_contract import PreflightPolicyContract


def test_hardware_inspector():
    info = HardwareInspector.inspect()
    assert "cpu_count_physical" in info
    assert "system_ram_total_gb" in info
    assert info["system_ram_total_gb"] > 0
    assert "torch_version" in info
    assert isinstance(info["cuda_available"], bool)


def test_memory_estimator():
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset", max_seq_length=2048),
        hardware=HardwareConfig(per_device_train_batch_size=2, mixed_precision="bf16"),
    )
    est = MemoryEstimator.estimate(cfg, total_params=1_000_000_000)
    assert est["total_params"] == 1_000_000_000
    assert est["weights_mb"] == 1907.35  # ~1.9 GB in 2 bytes/param
    assert est["optimizer_mb"] == 7629.39  # ~7.6 GB in 8 bytes/param
    assert est["estimated_total_gb"] > 0


def test_empirical_probe_on_micro_llama(micro_llama_model):
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=64),
        hardware=HardwareConfig(per_device_train_batch_size=2, target_device="cpu"),
    )
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is True
    assert report["isolation_mode"] == "in_process"
    assert report["verdict"] in [
        "PASS_SAFE_HEADROOM",
        "WARNING_HIGH_MEMORY_PRESSURE",
        "PASS_NON_REPRESENTATIVE_HEADROOM",
        "PASS_CPU_ONLY_UNVERIFIED_GPU",
    ]
    assert report["measured_peak_allocated_mb"] > 0
    # CPU execution must truthfully state that discrete GPU fit is not guaranteed
    assert report["gpu_fit_guaranteed"] is False
    assert report["is_representative"] is False


def test_empirical_probe_subprocess_execution(micro_llama_model):
    """Verifies that the empirical probe executes safely inside an isolated subprocess."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )
    report = EmpiricalMemoryProbe.run_probe(
        model=micro_llama_model,
        config=cfg,
        in_subprocess=True,
        timeout_seconds=60,
    )
    assert report["probe_successful"] is True
    assert report["isolation_mode"] == "subprocess"
    assert report["verdict"] in [
        "PASS_SAFE_HEADROOM",
        "WARNING_HIGH_MEMORY_PRESSURE",
        "PASS_NON_REPRESENTATIVE_HEADROOM",
        "PASS_CPU_ONLY_UNVERIFIED_GPU",
    ]
    assert report["measured_peak_allocated_mb"] > 0
    assert report["gpu_fit_guaranteed"] is False
    assert report["is_representative"] is False


def test_empirical_probe_subprocess_timeout(micro_llama_model):
    """Verifies that the subprocess probe enforces timeout and returns PROBE_TIMEOUT verdict."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )
    report = EmpiricalMemoryProbe.run_probe(
        model=micro_llama_model,
        config=cfg,
        in_subprocess=True,
        timeout_seconds=0.001,  # Guaranteed instantaneous timeout
    )
    assert report["probe_successful"] is False
    assert report["verdict"] == "PROBE_TIMEOUT"
    assert "timed out" in report["error_detail"].lower()


def test_empirical_probe_subprocess_memory_limit(micro_llama_model):
    """Verifies that exceeding memory ceiling terminates subprocess cleanly."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )
    report = EmpiricalMemoryProbe.run_probe(
        model=micro_llama_model,
        config=cfg,
        in_subprocess=True,
        timeout_seconds=60,
        max_process_memory_mb=1,  # 1 MB threshold is guaranteed to be exceeded by Python runtime
    )
    assert report["probe_successful"] is False
    assert report["verdict"] == "PROCESS_MEMORY_LIMIT_EXCEEDED"
    assert "exceeded process memory ceiling" in report["error_detail"].lower()


def test_empirical_probe_unsupported_sequence_length(micro_llama_model):
    """M-3 Fix: When max_seq_length > max_position_embeddings, emit UNSUPPORTED_SEQUENCE_LENGTH with probe_successful=False."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=1024),  # 1024 > 512 max_pos
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is False
    assert report["verdict"] == "UNSUPPORTED_SEQUENCE_LENGTH"
    assert "exceeds model max_position_embeddings" in report["error_detail"]


def test_empirical_probe_lora_and_checkpointing_realism(micro_llama_model):
    """M-2 Fix: Probe wraps model with LoRA, mixed precision, and gradient checkpointing."""
    cfg = TrainingJobConfig(
        task_type=TaskType.LORA,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(
            per_device_train_batch_size=1,
            target_device="cpu",
            gradient_checkpointing=True,
            mixed_precision="bf16",
        ),
    )
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is True
    assert report["measured_peak_allocated_mb"] > 0
    assert report["verdict"] in [
        "PASS_SAFE_HEADROOM",
        "WARNING_HIGH_MEMORY_PRESSURE",
        "PASS_NON_REPRESENTATIVE_HEADROOM",
        "PASS_CPU_ONLY_UNVERIFIED_GPU",
    ]


def test_empirical_probe_cpu_only_unverified_gpu_verdict(micro_llama_model):
    """Verifies that requesting CUDA when only running on CPU truthfully produces PASS_CPU_ONLY_UNVERIFIED_GPU."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cuda"),
    )
    report = EmpiricalMemoryProbe.run_probe(
        model=micro_llama_model,
        config=cfg,
        device=torch.device("cpu"),
    )
    assert report["probe_successful"] is True
    assert report["verdict"] == "PASS_CPU_ONLY_UNVERIFIED_GPU"
    assert report["gpu_fit_guaranteed"] is False
    assert report["is_representative"] is False
    assert len(report["representativeness_warnings"]) > 0


def test_empirical_probe_dpo_with_lora(micro_llama_model):
    """Verifies DPO memory probe with LoRA adapter disabling for reference forward pass."""
    cfg = TrainingJobConfig(
        task_type=TaskType.DPO,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
        adaptation_mode="lora",
    )
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is True
    assert report["verdict"] in [
        "PASS_SAFE_HEADROOM",
        "WARNING_HIGH_MEMORY_PRESSURE",
        "PASS_NON_REPRESENTATIVE_HEADROOM",
        "PASS_CPU_ONLY_UNVERIFIED_GPU",
    ]
    assert report["measured_peak_allocated_mb"] > 0


def test_empirical_probe_dpo_full_finetune(micro_llama_model):
    """Verifies DPO probe for full fine-tuning with duplicated frozen reference model."""
    cfg = TrainingJobConfig(
        task_type=TaskType.DPO,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )
    # Model without LoRA triggers full fine-tuning DPO reference model duplication
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is True
    assert report["measured_peak_allocated_mb"] > 0


def test_empirical_probe_setup_failure_rejection(monkeypatch, micro_llama_model):
    """Verifies that probe setup errors fail with UNSUPPORTED_CONFIGURATION instead of silent success."""
    cfg = TrainingJobConfig(
        task_type=TaskType.DPO,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
    )

    def mock_deepcopy_fail(obj):
        raise RuntimeError("Simulated out-of-memory during reference model deepcopy")

    monkeypatch.setattr(copy, "deepcopy", mock_deepcopy_fail)
    report = EmpiricalMemoryProbe.run_probe(micro_llama_model, cfg)
    assert report["probe_successful"] is False
    assert report["verdict"] == "UNSUPPORTED_CONFIGURATION"
    assert "setup failed" in report["error_detail"].lower()


def test_aggregate_rss_memory_sampling():
    """Verifies psutil aggregate memory sampling across parent and children."""
    current_proc = psutil.Process()
    rss = _get_aggregate_rss_mb(current_proc)
    assert rss > 0.0


def test_tokenizer_vocabulary_audit(micro_llama_model, micro_tokenizer):
    """Verifies vocabulary alignment and mismatch detection."""
    # 1. Aligned audit
    res_aligned = TokenizerInspector.audit_vocabulary(micro_llama_model, micro_tokenizer)
    assert res_aligned["model_vocab_size"] == 256
    assert res_aligned["is_safe"] is True
    assert res_aligned["verdict"] == "PASS_ALIGNED"

    # 2. Embedding overflow risk (tokenizer vocab > model vocab)
    small_model_config = LlamaConfig(vocab_size=100)
    res_overflow = TokenizerInspector.audit_vocabulary(small_model_config, micro_tokenizer)
    assert res_overflow["is_safe"] is False
    assert res_overflow["verdict"] == "FAIL_EMBEDDING_OVERFLOW"
    assert len(res_overflow["warnings"]) > 0
    assert "will cause fatal CUDA" in res_overflow["warnings"][0]

    # 3. Model vocab larger than tokenizer vocab
    large_model_config = LlamaConfig(vocab_size=512)
    res_large = TokenizerInspector.audit_vocabulary(large_model_config, micro_tokenizer)
    assert res_large["is_safe"] is True
    assert res_large["verdict"] == "WARNING_UNMAPPED_EMBEDDINGS"


def test_tokenizer_special_tokens_audit(micro_tokenizer):
    """Verifies audit of special tokens, pad token definition, and pad/eos collision."""
    res = TokenizerInspector.audit_special_tokens(micro_tokenizer)
    assert res["pad_token_defined"] is True
    assert res["special_tokens"]["pad_token"]["id"] == 0
    assert res["special_tokens"]["eos_token"]["id"] == 2
    assert res["pad_equals_eos"] is False
    assert res["verdict"] == "PASS_SAFE"

    # Test missing pad token
    class MockTokenizerNoPad:
        bos_token = "[BOS]"
        bos_token_id = 1
        eos_token = "[EOS]"
        eos_token_id = 2
        pad_token = None
        pad_token_id = None
        unk_token = "[UNK]"
        unk_token_id = 3

    res_no_pad = TokenizerInspector.audit_special_tokens(MockTokenizerNoPad())
    assert res_no_pad["pad_token_defined"] is False
    assert res_no_pad["verdict"] == "FAIL_MISSING_PAD_TOKEN"
    assert any("pad_token is NOT defined" in w for w in res_no_pad["warnings"])

    # Test pad equals eos collision
    class MockTokenizerPadCollision:
        bos_token = "[BOS]"
        bos_token_id = 1
        eos_token = "<|im_end|>"
        eos_token_id = 2
        pad_token = "<|im_end|>"
        pad_token_id = 2
        unk_token = "[UNK]"
        unk_token_id = 3

    res_collision = TokenizerInspector.audit_special_tokens(MockTokenizerPadCollision())
    assert res_collision["pad_token_defined"] is True
    assert res_collision["pad_equals_eos"] is True
    assert res_collision["verdict"] == "PASS_WITH_COLLISION_NOTICE"


def test_data_token_distribution(micro_tokenizer):
    """Verifies token distribution statistics, truncation counting, and padding ratio."""
    samples = [
        "User: Explain training.",
        "Continued pretraining processes raw domain text across deep architectures.",
        "Direct preference optimization aligns policies with human feedback reliably.",
        "Short text.",
    ]
    dist = DataInspector.analyze_token_distribution(
        tokenizer=micro_tokenizer,
        samples=samples,
        max_seq_length=5,
    )
    assert dist["total_samples"] == 4
    assert dist["min_length"] > 0
    assert dist["max_length"] >= dist["min_length"]
    assert "mean_length" in dist
    assert "p95" in dist
    assert dist["truncated_count"] > 0
    assert dist["truncated_percent"] > 0.0
    assert dist["verdict"] in ["PASS_CLEAN_DISTRIBUTION", "WARNING_HIGH_TRUNCATION"]

    # Empty sample list handling
    empty_dist = DataInspector.analyze_token_distribution(micro_tokenizer, samples=[])
    assert empty_dist["verdict"] == "ERROR_EMPTY_DATASET"


def test_preflight_inspection_suite(micro_llama_model, micro_tokenizer):
    """Verifies unified inspection suite combining vocabulary, special tokens, and distribution."""
    samples = ["Sample sentence one.", "Sample sentence two."]
    report = PreflightInspectionSuite.run_full_inspection(
        model_or_config=micro_llama_model,
        tokenizer=micro_tokenizer,
        samples=samples,
        max_seq_length=64,
    )
    assert report["overall_status"] in ["PASSED_CLEAN", "PASSED_WITH_WARNINGS"]
    assert "vocabulary_audit" in report
    assert "special_tokens_audit" in report
    assert "distribution_audit" in report
    assert report["vocabulary_audit"]["is_safe"] is True


def test_resource_classifier():
    """Verifies Hub ID, local path, and remote URI classification and jail enforcement."""
    # 1. Hub IDs
    id_hub1 = ResourceClassifier.classify("HuggingFaceTB/SmolLM-135M")
    assert id_hub1.kind == ResourceKind.HUB_ID
    assert id_hub1.is_hub is True
    assert id_hub1.is_local is False
    assert id_hub1.repo_id == "HuggingFaceTB/SmolLM-135M"

    id_hub2 = ResourceClassifier.classify("gpt2")
    assert id_hub2.kind == ResourceKind.HUB_ID
    assert id_hub2.is_hub is True

    id_hub3 = ResourceClassifier.classify("meta-llama/Meta-Llama-3-8B-Instruct")
    assert id_hub3.kind == ResourceKind.HUB_ID
    assert id_hub3.is_hub is True

    # 2. Local relative paths with path indicators
    id_local_rel = ResourceClassifier.classify("./data/dataset.jsonl")
    assert id_local_rel.kind == ResourceKind.LOCAL_PATH
    assert id_local_rel.is_local is True
    assert id_local_rel.is_hub is False

    # 3. Local absolute path
    test_file_path = Path(__file__).resolve().as_posix()
    id_local_abs = ResourceClassifier.classify(test_file_path)
    assert id_local_abs.kind == ResourceKind.LOCAL_PATH
    assert id_local_abs.is_local is True
    assert id_local_abs.exists_locally is True

    # 4. Non-existent path allowed for output directories
    id_output_dir = ResourceClassifier.classify("runs/experiment_01", allow_nonexistent_local=True)
    assert id_output_dir.kind == ResourceKind.LOCAL_PATH
    assert id_output_dir.is_local is True

    # 5. Remote URIs
    id_remote_http = ResourceClassifier.classify("https://huggingface.co/models/test")
    assert id_remote_http.kind == ResourceKind.REMOTE_URI
    assert id_remote_http.is_remote is True
    assert id_remote_http.is_local is False
    assert id_remote_http.is_hub is False

    id_remote_s3 = ResourceClassifier.classify("s3://bucket/models")
    assert id_remote_s3.kind == ResourceKind.REMOTE_URI
    assert id_remote_s3.is_remote is True

    # 6. Invalid / empty identifier
    id_empty = ResourceClassifier.classify("")
    assert id_empty.kind == ResourceKind.INVALID
    assert id_empty.validation_error is not None

    # 7. Path Jail validation
    allowed_root = Path(__file__).resolve().parent
    # Hub IDs automatically pass local root check
    is_valid, err = ResourceClassifier.validate_in_roots(id_hub1, [allowed_root])
    assert is_valid is True
    assert err is None

    # Local file inside root passes
    is_valid_local, err_local = ResourceClassifier.validate_in_roots(id_local_abs, [allowed_root.parent])
    assert is_valid_local is True

    # Remote URI fails root validation
    is_valid_remote, err_remote = ResourceClassifier.validate_in_roots(id_remote_http, [allowed_root])
    assert is_valid_remote is False
    assert "Remote URI" in err_remote


def test_preflight_policy_contract(tmp_path):
    """Verifies manager-facing PreflightPolicyContract methods."""
    # 1. Resource classification
    res_id = PreflightPolicyContract.classify_resource("HuggingFaceTB/SmolLM-135M")
    assert res_id.kind == ResourceKind.HUB_ID
    assert res_id.is_hub is True

    # 2. Resource validation in roots
    valid, err = PreflightPolicyContract.validate_resource_in_roots(
        identifier="HuggingFaceTB/SmolLM-135M",
        allowed_roots=[tmp_path],
        field_name="model_name_or_path",
        allow_hub_ids=True,
    )
    assert valid is True
    assert err is None

    # Disallow Hub ID when requested
    valid_no_hub, err_no_hub = PreflightPolicyContract.validate_resource_in_roots(
        identifier="HuggingFaceTB/SmolLM-135M",
        allowed_roots=[tmp_path],
        field_name="model_name_or_path",
        allow_hub_ids=False,
    )
    assert valid_no_hub is False
    assert "does not permit Hub identifiers" in err_no_hub

    # 3. Hardware inspection
    hw = PreflightPolicyContract.inspect_hardware()
    assert "cpu_count_physical" in hw
    assert "system_ram_total_gb" in hw

    # 4. Static memory estimation
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="test/model"),
        dataset=DatasetConfig(dataset_name_or_path="test/dataset", max_seq_length=512),
        hardware=HardwareConfig(per_device_train_batch_size=1, mixed_precision="fp16"),
    )
    est = PreflightPolicyContract.estimate_memory(cfg, total_params=100_000_000)
    assert est["total_params"] == 100_000_000
    assert est["estimated_total_gb"] > 0


def test_isolated_tokenizer_inspection_suite(micro_llama_model, micro_tokenizer, tmp_path):
    """Verifies isolated subprocess tokenizer inspection execution and budget limits."""
    # Save model and tokenizer to tmp_path
    micro_llama_model.save_pretrained(str(tmp_path))
    micro_tokenizer.save_pretrained(str(tmp_path))

    # 1. Clean run in isolated subprocess
    res = PreflightPolicyContract.audit_tokenizer_isolated(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        timeout_seconds=30,
        max_process_memory_mb=4096,
    )
    assert res["overall_status"] in ["PASSED_CLEAN", "PASSED_WITH_WARNINGS"]
    assert "vocabulary_audit" in res
    assert res["vocabulary_audit"]["is_safe"] is True
    assert res["measured_peak_memory_mb"] > 0

    # 2. Subprocess timeout enforcement
    timeout_res = PreflightPolicyContract.audit_tokenizer_isolated(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        timeout_seconds=0.001,
    )
    assert timeout_res["overall_status"] == "CRITICAL_ISSUES_FOUND"
    assert timeout_res["verdict"] == "INSPECTION_TIMEOUT"
    assert "timed out" in timeout_res["error_detail"].lower()

    # 3. Subprocess memory limit enforcement
    mem_res = PreflightPolicyContract.audit_tokenizer_isolated(
        model_path=str(tmp_path),
        tokenizer_path=str(tmp_path),
        max_process_memory_mb=1,  # 1 MB threshold is guaranteed to be exceeded
    )
    assert mem_res["overall_status"] == "CRITICAL_ISSUES_FOUND"
    assert mem_res["verdict"] == "PROCESS_MEMORY_LIMIT_EXCEEDED"


def test_bounded_stream_drainer_hostile_no_newline():
    """Verifies that BoundedStreamDrainer reads hostile no-newline streams in fixed-size chunks
    without unbounded memory allocation and enforces max_bytes ceiling strictly.
    """
    import io
    from llm_training_stack.preflight.probe import BoundedStreamDrainer

    # 100 KB payload with zero newlines
    hostile_payload = "X" * 102400
    stream = io.StringIO(hostile_payload)

    drainer = BoundedStreamDrainer(stream, max_bytes=32768)
    out = drainer.get_output(timeout_sec=2.0)

    assert len(out) == 32768
    assert drainer.bytes_read == 32768


def test_bounded_stream_drainer_subprocess_flood():
    """CEO Delta 12: Verifies BoundedStreamDrainer on actual child subprocess emitting heavy spam
    and huge lines without newlines on both stdout and stderr, avoiding parent deadlock.
    """
    import subprocess
    import sys
    from llm_training_stack.preflight.probe import BoundedStreamDrainer

    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('A' * 131072); sys.stdout.flush(); sys.stderr.write(('B' * 1024 + '\\n') * 64); sys.stderr.flush()"
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_drainer = BoundedStreamDrainer(proc.stdout, max_bytes=32768)
    stderr_drainer = BoundedStreamDrainer(proc.stderr, max_bytes=32768)

    proc.wait(timeout=10.0)
    assert proc.returncode == 0

    stdout_out = stdout_drainer.get_output(timeout_sec=2.0)
    stderr_out = stderr_drainer.get_output(timeout_sec=2.0)

    assert len(stdout_out) == 32768
    assert stdout_drainer.bytes_read == 32768
    assert len(stderr_out) == 32768
    assert stderr_drainer.bytes_read == 32768


def test_empirical_probe_precision_and_optimizer_rejections(micro_llama_model):
    """CEO Delta 10/12 & Research Handoff: Verifies CPU FP16, fused optimizer, and invalid optimizer rejections."""
    # 1. FP16 on CPU rejected
    cfg_fp16 = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu", mixed_precision="fp16"),
    )
    rep_fp16 = EmpiricalMemoryProbe.run_probe(model=micro_llama_model, config=cfg_fp16, device=torch.device("cpu"))
    assert rep_fp16["probe_successful"] is False
    assert rep_fp16["verdict"] == "UNSUPPORTED_CONFIGURATION"
    assert "FP16 mixed precision is not supported on CPU execution" in rep_fp16["error_detail"]
    assert rep_fp16["gpu_fit_guaranteed"] is False

    # 2. adamw_torch_fused on CPU rejected
    cfg_fused = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
        hardware=HardwareConfig(per_device_train_batch_size=1, target_device="cpu"),
        optimizer=OptimizerConfig(optimizer_type="adamw_torch_fused"),
    )
    rep_fused = EmpiricalMemoryProbe.run_probe(model=micro_llama_model, config=cfg_fused, device=torch.device("cpu"))
    assert rep_fused["probe_successful"] is False
    assert rep_fused["verdict"] == "UNSUPPORTED_CONFIGURATION"
    assert "adamw_torch_fused optimizer requires a CUDA device" in rep_fused["error_detail"]
    assert rep_fused["gpu_fit_guaranteed"] is False

    # 3. Invalid optimizer type rejected at schema level and probe level
    with pytest.raises(Exception):
        OptimizerConfig(optimizer_type="invalid_opt")

    cfg_inv = cfg_fused.model_copy(deep=True)
    object.__setattr__(cfg_inv.optimizer, "optimizer_type", "invalid_opt")
    rep_inv = EmpiricalMemoryProbe.run_probe(model=micro_llama_model, config=cfg_inv, device=torch.device("cpu"))
    assert rep_inv["probe_successful"] is False
    assert rep_inv["verdict"] == "UNSUPPORTED_CONFIGURATION"
    assert "Unsupported optimizer type" in rep_inv["error_detail"]
    assert rep_inv["gpu_fit_guaranteed"] is False


def test_preflight_policy_contract_budget_clamp_rejections(tmp_path):
    """CEO Delta 12 & Research Handoff: PreflightPolicyContract budget parameters reject non-positive values."""
    cfg = TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(model_name_or_path="mock"),
        dataset=DatasetConfig(dataset_name_or_path="mock", max_seq_length=32),
    )
    # 1. Negative timeout
    with pytest.raises(ValueError, match="Operator policy violation.*timeout_seconds.*must be strictly positive"):
        PreflightPolicyContract.run_isolated_probe(model_path=str(tmp_path), config=cfg, timeout_seconds=-5)

    # 2. Zero memory ceiling
    with pytest.raises(ValueError, match="Operator policy violation.*max_process_memory_mb.*must be strictly positive"):
        PreflightPolicyContract.run_isolated_probe(model_path=str(tmp_path), config=cfg, max_process_memory_mb=0)

    # 3. Negative inspection timeout
    with pytest.raises(ValueError, match="Operator policy violation.*timeout_seconds.*must be strictly positive"):
        PreflightPolicyContract.audit_tokenizer_isolated(model_path=str(tmp_path), timeout_seconds=-1)

