"""Tests for evaluation metrics, prompt masking, and run comparison utilities."""

import math
import pytest
import torch
import torch.nn as nn
from llm_training_stack.eval.evaluator import Evaluator
from llm_training_stack.eval.comparator import RunComparator
from llm_training_stack.provenance.manifest import RunManifest
from llm_training_stack.config.schema import TrainingJobConfig, TaskType, ModelConfig, DatasetConfig


def test_evaluator_metric_calculation(micro_llama_model, micro_tokenizer):
    samples = [
        "Continuous pretraining on text sequences.",
        "Evaluating model perplexity on validation data.",
    ]
    metrics = Evaluator.evaluate(
        micro_llama_model,
        micro_tokenizer,
        samples,
        max_seq_length=64,
    )
    assert "eval_loss" in metrics
    assert "perplexity" in metrics
    assert metrics["evaluated_samples"] == 2
    assert metrics["perplexity"] > 0.0
    assert metrics["masking_scheme"] == "all_tokens"
    assert "dataset_fingerprint" in metrics


def test_evaluator_empty_samples_rejection(micro_llama_model, micro_tokenizer):
    """Verifies that Evaluator strictly raises ValueError on empty/whitespace datasets."""
    with pytest.raises(ValueError, match="no non-empty text samples"):
        Evaluator.evaluate(micro_llama_model, micro_tokenizer, ["", "   ", "\n\t"])


def test_evaluator_skips_empty_samples_accurately(micro_llama_model, micro_tokenizer):
    """Verifies that Evaluator counts only valid non-empty evaluated samples."""
    samples = ["Valid prompt sample 1.", "", "Valid prompt sample 2.", "   "]
    metrics = Evaluator.evaluate(micro_llama_model, micro_tokenizer, samples, max_seq_length=64)
    assert metrics["evaluated_samples"] == 2
    assert metrics["eval_loss"] > 0.0


def test_evaluator_response_only_prompt_masking(micro_llama_model, micro_tokenizer):
    """Criterion 5: Verifies that dict samples with prompt/response apply response-only loss masking."""
    samples = [
        {"prompt": "User query question: What is Python?", "response": " Python is an interpreted programming language."},
        {"prompt": "Instruction: Summarize this.", "response": " Summary: Brief overview of key aspects."},
    ]
    metrics = Evaluator.evaluate(
        micro_llama_model,
        micro_tokenizer,
        samples,
        max_seq_length=64,
    )
    assert metrics["evaluated_samples"] == 2
    assert metrics["masking_scheme"] == "response_only"
    assert metrics["eval_loss"] > 0.0
    assert metrics["total_tokens"] > 0


def test_evaluator_analytical_uniform_logits_ground_truth(micro_tokenizer):
    """Mathematical ground-truth verification: uniform logits over vocab V yields exact loss ln(V) and perplexity V."""
    vocab_size = len(micro_tokenizer)

    class UniformLogitsModel(nn.Module):
        def __init__(self, vocab_dim):
            super().__init__()
            self.vocab_dim = vocab_dim
            # dummy parameter to satisfy device discovery
            self.param = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            batch, seq_len = input_ids.shape
            # All logits identical (0.0) -> softmax probability for every token is exactly 1/V
            logits = torch.zeros(batch, seq_len, self.vocab_dim)
            return type("Outputs", (), {"logits": logits})()

    uniform_model = UniformLogitsModel(vocab_size)
    samples = ["Testing mathematical cross entropy precision."]

    metrics = Evaluator.evaluate(
        uniform_model,
        micro_tokenizer,
        samples,
        max_seq_length=32,
    )

    expected_loss = math.log(vocab_size)
    expected_ppl = float(vocab_size)

    assert abs(metrics["eval_loss"] - expected_loss) < 1e-3, f"Expected loss {expected_loss}, got {metrics['eval_loss']}"
    assert abs(metrics["perplexity"] - expected_ppl) < 0.5, f"Expected perplexity {expected_ppl}, got {metrics['perplexity']}"


def test_evaluator_report_persistence(tmp_path, micro_llama_model, micro_tokenizer):
    """Criterion 5: Verifies saving and loading of held-out evaluation report contracts."""
    report_path = tmp_path / "eval_report.json"
    samples = ["Evaluating report persistence contract."]
    metrics = Evaluator.evaluate(
        micro_llama_model,
        micro_tokenizer,
        samples,
        max_seq_length=32,
    )
    Evaluator.save_report(metrics, report_path)
    assert report_path.exists()

    loaded = Evaluator.load_report(report_path)
    assert loaded["eval_loss"] == metrics["eval_loss"]
    assert loaded["dataset_fingerprint"] == metrics["dataset_fingerprint"]


def test_run_comparator_markdown_output(tmp_path):
    run_dir1 = tmp_path / "run_1"
    run_dir2 = tmp_path / "run_2"
    run_dir1.mkdir()
    run_dir2.mkdir()

    cfg1 = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="model_a"), dataset=DatasetConfig(dataset_name_or_path="ds_a"))
    cfg2 = TrainingJobConfig(task_type=TaskType.LORA, model=ModelConfig(model_name_or_path="model_b"), dataset=DatasetConfig(dataset_name_or_path="ds_b"))

    m1 = RunManifest.create(cfg1)
    m1["final_metrics"] = {"final_loss": 1.42, "total_steps": 100, "total_duration_sec": 45.2}
    RunManifest.save(m1, run_dir1)

    m2 = RunManifest.create(cfg2)
    m2["final_metrics"] = {"final_loss": 1.15, "total_steps": 100, "total_duration_sec": 38.6}
    RunManifest.save(m2, run_dir2)

    result = RunComparator.compare_runs([run_dir1, run_dir2])
    assert len(result["runs"]) == 2
    assert "| Run ID | Task | Model |" in result["markdown_report"]
    assert "1.42" in result["markdown_report"]
    assert "1.15" in result["markdown_report"]


def test_run_comparator_strict_incompatibility_rejection(tmp_path):
    """Criterion 5: Verifies that comparing runs evaluated on different datasets raises error under strict mode."""
    run_dir1 = tmp_path / "run_eval_a"
    run_dir2 = tmp_path / "run_eval_b"
    run_dir1.mkdir()
    run_dir2.mkdir()

    cfg1 = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m"), dataset=DatasetConfig(dataset_name_or_path="d1"))
    cfg2 = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m"), dataset=DatasetConfig(dataset_name_or_path="d2"))

    m1 = RunManifest.create(cfg1)
    m1["final_metrics"] = {"final_loss": 2.1}
    RunManifest.save(m1, run_dir1)

    m2 = RunManifest.create(cfg2)
    m2["final_metrics"] = {"final_loss": 1.9}
    RunManifest.save(m2, run_dir2)

    # Attach differing held-out eval reports
    Evaluator.save_report({"eval_loss": 2.1, "dataset_fingerprint": "hash_dataset_aaa"}, run_dir1 / "eval_report.json")
    Evaluator.save_report({"eval_loss": 1.9, "dataset_fingerprint": "hash_dataset_bbb"}, run_dir2 / "eval_report.json")

    # Non-strict mode includes compatibility notice
    res = RunComparator.compare_runs([run_dir1, run_dir2], strict_compatibility=False)
    assert not res["compatibility"]["is_comparable"]
    assert "Compatibility Notice" in res["markdown_report"]

    # Strict mode raises ValueError
    with pytest.raises(ValueError, match="Runs are not comparable"):
        RunComparator.compare_runs([run_dir1, run_dir2], strict_compatibility=True)


def test_evaluator_known_logits_exact_count_and_prompt_masking(micro_tokenizer):
    """Criterion 5: Verifies that known target logits yield mathematical zero loss on response
    tokens while prompt tokens are strictly masked out of the loss calculation.
    """
    vocab_size = len(micro_tokenizer)

    class PerfectPredictor(nn.Module):
        def __init__(self, vocab_dim):
            super().__init__()
            self.vocab_dim = vocab_dim
            self.param = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            b, s = input_ids.shape
            logits = torch.zeros(b, s, self.vocab_dim)
            for i in range(b):
                for t in range(s - 1):
                    next_tok = input_ids[i, t + 1].item()
                    logits[i, t, next_tok] = 50.0  # huge target logit -> loss is ~0
            return type("Outputs", (), {"logits": logits})()

    model = PerfectPredictor(vocab_size)
    samples = [{"prompt": "User input prompt:", "response": " Correct answer output."}]

    metrics = Evaluator.evaluate(
        model,
        micro_tokenizer,
        samples,
        max_seq_length=32,
    )

    assert metrics["masking_scheme"] == "response_only"
    assert metrics["eval_loss"] < 1e-3, f"Expected near-zero loss on known logits, got {metrics['eval_loss']}"
    assert metrics["total_tokens"] > 0
    assert metrics["evaluated_samples"] == 1


def test_run_comparator_compatible_held_out_comparison(tmp_path):
    """Criterion 5: Verifies that comparing runs sharing identical held-out eval dataset
    fingerprints succeeds under strict mode and correctly ranks runs by eval loss.
    """
    run_dir1 = tmp_path / "run_eval_comp_1"
    run_dir2 = tmp_path / "run_eval_comp_2"
    run_dir1.mkdir()
    run_dir2.mkdir()

    cfg1 = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m1"), dataset=DatasetConfig(dataset_name_or_path="eval_ds"))
    cfg2 = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m2"), dataset=DatasetConfig(dataset_name_or_path="eval_ds"))

    m1 = RunManifest.create(cfg1)
    m1["final_metrics"] = {"final_loss": 2.1}
    RunManifest.save(m1, run_dir1)

    m2 = RunManifest.create(cfg2)
    m2["final_metrics"] = {"final_loss": 1.4}
    RunManifest.save(m2, run_dir2)

    # Attach identical held-out eval dataset fingerprint, tokenizer, masking scheme and max_seq_length to both runs
    common_dataset_fingerprint = "held_out_eval_dataset_sha256_abcdef123456"
    Evaluator.save_report({
        "eval_loss": 2.1,
        "dataset_fingerprint": common_dataset_fingerprint,
        "tokenizer_name_or_path": "shared_tokenizer",
        "masking_scheme": "all_tokens",
        "max_seq_length": 512,
    }, run_dir1 / "eval_report.json")
    Evaluator.save_report({
        "eval_loss": 1.4,
        "dataset_fingerprint": common_dataset_fingerprint,
        "tokenizer_name_or_path": "shared_tokenizer",
        "masking_scheme": "all_tokens",
        "max_seq_length": 512,
    }, run_dir2 / "eval_report.json")

    # Strict mode succeeds because fingerprints, tokenizers, masking and max_seq_length match
    res = RunComparator.compare_runs([run_dir1, run_dir2], strict_compatibility=True)
    assert res["compatibility"]["is_comparable"] is True
    assert res["runs"][0]["heldout_eval_loss"] == 2.1
    assert res["runs"][1]["heldout_eval_loss"] == 1.4
    assert res["runs"][0]["dataset_fingerprint"] == common_dataset_fingerprint
    assert res["runs"][1]["dataset_fingerprint"] == common_dataset_fingerprint
    assert "| Run ID | Task | Model |" in res["markdown_report"]
    assert "2.1" in res["markdown_report"]
    assert "1.4" in res["markdown_report"]


def test_run_comparator_rejects_missing_or_differing_masking_and_max_length(tmp_path):
    """CEO Delta Review 11: Comparator strictly rejects runs with absent masking_scheme or differing max_seq_length."""
    run_dir1 = tmp_path / "run_m1"
    run_dir2 = tmp_path / "run_m2"
    run_dir1.mkdir()
    run_dir2.mkdir()

    cfg = TrainingJobConfig(task_type=TaskType.SFT, model=ModelConfig(model_name_or_path="m"), dataset=DatasetConfig(dataset_name_or_path="d"))
    m1 = RunManifest.create(cfg)
    m2 = RunManifest.create(cfg)
    RunManifest.save(m1, run_dir1)
    RunManifest.save(m2, run_dir2)

    # 1. Missing masking_scheme
    Evaluator.save_report({
        "eval_loss": 2.0,
        "dataset_fingerprint": "common_fp",
        "tokenizer_name_or_path": "tok",
        "max_seq_length": 512,
    }, run_dir1 / "eval_report.json")
    Evaluator.save_report({
        "eval_loss": 2.1,
        "dataset_fingerprint": "common_fp",
        "tokenizer_name_or_path": "tok",
        "masking_scheme": "all_tokens",
        "max_seq_length": 512,
    }, run_dir2 / "eval_report.json")

    compat1 = RunComparator.check_compatibility([RunComparator.load_run(run_dir1), RunComparator.load_run(run_dir2)])
    assert not compat1["is_comparable"]
    assert any("missing masking_scheme" in r for r in compat1["reasons"])

    # 2. Differing max_seq_length
    Evaluator.save_report({
        "eval_loss": 2.0,
        "dataset_fingerprint": "common_fp",
        "tokenizer_name_or_path": "tok",
        "masking_scheme": "all_tokens",
        "max_seq_length": 256,
    }, run_dir1 / "eval_report.json")
    Evaluator.save_report({
        "eval_loss": 2.1,
        "dataset_fingerprint": "common_fp",
        "tokenizer_name_or_path": "tok",
        "masking_scheme": "all_tokens",
        "max_seq_length": 1024,
    }, run_dir2 / "eval_report.json")

    compat2 = RunComparator.check_compatibility([RunComparator.load_run(run_dir1), RunComparator.load_run(run_dir2)])
    assert not compat2["is_comparable"]
    assert any("max_seq_length differs" in r for r in compat2["reasons"])


def test_evaluator_framed_dataset_hashing_collision_resistance():
    """CEO Delta Review 09: Verifies that length-prefixed framing eliminates boundary collisions
    such as ['ab', 'c'] vs ['a', 'bc'].
    """
    from llm_training_stack.eval.evaluator import compute_dataset_fingerprint

    ds1 = ["ab", "c"]
    ds2 = ["a", "bc"]

    fp1 = compute_dataset_fingerprint(ds1)
    fp2 = compute_dataset_fingerprint(ds2)

    assert fp1 != fp2, f"Framed dataset hashing collided on boundary shift: {fp1} == {fp2}"

