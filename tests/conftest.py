"""Pytest fixtures providing lightweight micro-models and temporary environments."""

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


@pytest.fixture(scope="session")
def micro_llama_config():
    """Ultra-compact 2-layer LLaMA model configuration for instant CPU testing."""
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture(scope="function")
def micro_llama_model(micro_llama_config):
    """Instantiated 2-layer LLaMA model on CPU (fresh instance per test)."""
    torch.manual_seed(42)
    model = LlamaForCausalLM(micro_llama_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def micro_tokenizer(tmp_path_factory):
    """Minimal fast tokenizer matching vocab_size=256."""
    tok_dir = tmp_path_factory.mktemp("tokenizer")
    raw_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], vocab_size=256)

    # Train on small corpus
    corpus = [
        "User: Explain training.\nAssistant: Training updates model parameters.\n",
        "Continued pretraining processes raw domain text.\n",
        "Direct preference optimization aligns policies with human feedback.\n",
    ]
    raw_tokenizer.train_from_iterator(corpus, trainer=trainer)
    current_vocab_len = len(raw_tokenizer.get_vocab())
    if current_vocab_len < 256:
        raw_tokenizer.add_tokens([f"<dummy_{i}>" for i in range(current_vocab_len, 256)])
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
    fast_tokenizer.pad_token = "[PAD]"
    fast_tokenizer.bos_token = "[BOS]"
    fast_tokenizer.eos_token = "[EOS]"
    fast_tokenizer.unk_token = "[UNK]"
    fast_tokenizer.pad_token_id = 0
    fast_tokenizer.bos_token_id = 1
    fast_tokenizer.eos_token_id = 2

    tok_file = tok_dir / "tokenizer.json"
    fast_tokenizer.save_pretrained(str(tok_dir))
    return fast_tokenizer


@pytest.fixture
def base_test_config(tmp_path):
    from llm_training_stack.config.schema import (
        TrainingJobConfig,
        TaskType,
        ModelConfig,
        DatasetConfig,
        HardwareConfig,
        LoggingConfig,
    )
    return TrainingJobConfig(
        task_type=TaskType.SFT,
        model=ModelConfig(
            model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            torch_dtype="float32",
        ),
        dataset=DatasetConfig(
            dataset_name_or_path="synthetic",
            max_seq_length=64,
            train_sample_limit=10,
        ),
        hardware=HardwareConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            target_device="cpu",
        ),
        logging=LoggingConfig(
            output_dir=str(tmp_path / "pipeline_run"),
            save_steps=5,
        ),
        max_steps=4,
    )

