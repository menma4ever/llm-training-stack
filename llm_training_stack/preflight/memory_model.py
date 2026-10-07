"""Analytical memory estimation models for LLM training."""

from typing import Dict, Any, Optional
from llm_training_stack.config.schema import TrainingJobConfig, TaskType


class MemoryEstimator:
    """Computes transparent, analytical memory footprints with explicit assumptions."""

    @staticmethod
    def estimate(
        config: TrainingJobConfig,
        total_params: int,
        trainable_params: Optional[int] = None,
        hidden_size: int = 4096,
        num_layers: int = 32,
    ) -> Dict[str, Any]:
        """Calculates memory breakdown in bytes and gigabytes."""
        if trainable_params is None:
            if config.task_type == TaskType.LORA:
                # Typical LoRA adapters are ~0.1% to 1.0% of base model
                trainable_params = int(total_params * 0.005)
            else:
                trainable_params = total_params

        # Bytes per parameter based on precision
        if config.hardware.mixed_precision in ["fp16", "bf16"]:
            bytes_per_param = 2
        else:
            bytes_per_param = 4  # FP32

        # 1. Weights memory
        weights_bytes = total_params * bytes_per_param

        # 2. Gradients memory (only for trainable parameters)
        gradients_bytes = trainable_params * bytes_per_param

        # 3. Optimizer states memory (AdamW: 8 bytes per trainable param for m and v in FP32)
        optimizer_bytes_per_param = 8
        optimizer_bytes = trainable_params * optimizer_bytes_per_param

        # 4. Activation memory estimation
        batch_size = config.hardware.per_device_train_batch_size
        seq_len = config.dataset.max_seq_length

        # Rough activation model: ~34 * B * S * H * L bytes (standard)
        # Activation checkpointing reduces this by approximately 75%
        act_factor = 34 if not config.hardware.gradient_checkpointing else 8
        activations_bytes = batch_size * seq_len * hidden_size * num_layers * (act_factor / 16) * bytes_per_param

        # Framework base overhead buffer (CUDA contexts, kernels, allocators)
        framework_overhead_bytes = int(0.8 * (1024 ** 3))

        total_bytes = weights_bytes + gradients_bytes + optimizer_bytes + activations_bytes + framework_overhead_bytes

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percent": round((trainable_params / total_params) * 100, 3) if total_params > 0 else 0.0,
            "precision": config.hardware.mixed_precision,
            "bytes_per_param": bytes_per_param,
            "weights_mb": round(weights_bytes / (1024 ** 2), 2),
            "gradients_mb": round(gradients_bytes / (1024 ** 2), 2),
            "optimizer_mb": round(optimizer_bytes / (1024 ** 2), 2),
            "activations_mb": round(activations_bytes / (1024 ** 2), 2),
            "framework_overhead_mb": round(framework_overhead_bytes / (1024 ** 2), 2),
            "estimated_total_mb": round(total_bytes / (1024 ** 2), 2),
            "estimated_total_gb": round(total_bytes / (1024 ** 3), 2),
            "assumptions": [
                f"Weights in {config.hardware.mixed_precision.upper() or 'FP32'} ({bytes_per_param} bytes/param)",
                f"Gradients computed strictly for {trainable_params:,} trainable parameters",
                f"AdamW maintains 2 moments in FP32 (8 bytes/param) for trainable weights",
                f"Gradient checkpointing: {'ENABLED (reduced activation memory)' if config.hardware.gradient_checkpointing else 'DISABLED'}",
                f"Per-device micro-batch size = {batch_size}, max sequence length = {seq_len}",
                "Note: Static analytical models do not account for PyTorch caching allocator fragmentation or dynamic tensor resizing.",
            ]
        }
