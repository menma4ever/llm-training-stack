"""Bounded empirical memory probe for verifying actual hardware fit.

Hardened with isolated subprocess sandboxing, aggregate process memory ceilings,
realistic objective matching (CPT, SFT, orthogonal LoRA, DPO reference modeling),
ongoing bounded stream draining, and truthful GPU/CPU fit guarantees.
"""

import contextlib
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_training_stack.config.schema import TaskType, TrainingJobConfig
from llm_training_stack.preflight.hardware import HardwareInspector
from llm_training_stack.preflight.memory_model import MemoryEstimator


def _cleanup_process_tree(proc: subprocess.Popen, psutil_proc: Optional[psutil.Process], timeout_sec: float = 3.0):
    """Recursively terminates descendant processes, sends SIGKILL if necessary, and reaps zombies."""
    children = []
    if psutil_proc is not None:
        try:
            children = psutil_proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []

    for child in children:
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        proc.terminate()
    except Exception:
        pass

    procs_to_wait = [p for p in children if p.is_running()]
    if psutil_proc is not None and psutil_proc.is_running():
        procs_to_wait.append(psutil_proc)

    if procs_to_wait:
        try:
            gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout_sec)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def _get_aggregate_rss_mb(p: psutil.Process) -> float:
    """Calculates total RSS memory in MB across parent process and all active descendants."""
    total_rss = 0
    try:
        total_rss += p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        for child in p.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return round(total_rss / (1024 ** 2), 2)


class BoundedStreamDrainer:
    """Continuously drains an IO stream into a bounded memory buffer in a background thread,
    preventing OS pipe buffer exhaustion and child process deadlock.
    """
    def __init__(self, stream, max_bytes: int = 32768):
        self.stream = stream
        self.max_bytes = max_bytes
        self.buffer = []
        self.bytes_read = 0
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                # Read in bounded 4096-character/byte chunks to prevent unbounded readline() allocations
                if hasattr(self.stream, "read"):
                    chunk = self.stream.read(4096)
                elif hasattr(self.stream, "readline"):
                    chunk = self.stream.readline(4096)
                else:
                    break
                if not chunk:
                    break
                encoded = chunk.encode("utf-8", errors="replace") if isinstance(chunk, str) else chunk
                if self.bytes_read < self.max_bytes:
                    space_left = self.max_bytes - self.bytes_read
                    self.buffer.append(encoded[:space_left].decode("utf-8", errors="replace"))
                    self.bytes_read += min(len(encoded), space_left)
        except Exception:
            pass
        finally:
            try:
                self.stream.close()
            except Exception:
                pass

    def get_output(self, timeout_sec: float = 2.0) -> str:
        self.thread.join(timeout=timeout_sec)
        return "".join(self.buffer)


class EmpiricalMemoryProbe:
    """Executes a representative forward/backward probe to measure empirical peak memory."""

    DEFAULT_MEMORY_CEILING_MB = 4096

    @staticmethod
    def run_probe(
        model: Union[nn.Module, str, Path],
        config: TrainingJobConfig,
        device: Optional[torch.device] = None,
        in_subprocess: bool = False,
        timeout_seconds: float = 60,
        max_process_memory_mb: Optional[int] = 4096,
    ) -> Dict[str, Any]:
        """Runs the memory probe matching configured objective, precision, and PEFT adaptation.

        When in_subprocess=True, any CUDA OOM, segmentation fault, or allocator panic is
        isolated within a child OS process, preventing host termination.
        """
        memory_ceiling = max_process_memory_mb if max_process_memory_mb is not None else EmpiricalMemoryProbe.DEFAULT_MEMORY_CEILING_MB

        # If model is path/string or in_subprocess is requested, execute isolated subprocess probe
        if in_subprocess or isinstance(model, (str, Path)):
            if isinstance(model, nn.Module):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model.save_pretrained(tmp_dir)
                    return EmpiricalMemoryProbe.run_probe_subprocess(
                        model_path=tmp_dir,
                        config=config,
                        timeout_seconds=timeout_seconds,
                        max_process_memory_mb=memory_ceiling,
                        device=str(device) if device else None,
                    )
            else:
                return EmpiricalMemoryProbe.run_probe_subprocess(
                    model_path=model,
                    config=config,
                    timeout_seconds=timeout_seconds,
                    max_process_memory_mb=memory_ceiling,
                    device=str(device) if device else None,
                )

        # In-process probe execution
        if device is None:
            if config.hardware.target_device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif config.hardware.target_device == "cpu" or not torch.cuda.is_available():
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_size = config.hardware.per_device_train_batch_size
        max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)

        # 1. Sequence Length Contract: Fail probe on sequence length overflow
        if max_pos is not None and config.dataset.max_seq_length > max_pos:
            return {
                "probe_successful": False,
                "isolation_mode": "in_process",
                "verdict": "UNSUPPORTED_SEQUENCE_LENGTH",
                "device": str(device),
                "measurement_type": "none",
                "gpu_fit_guaranteed": False,
                "is_representative": False,
                "representativeness_warnings": [
                    f"Configured max_seq_length ({config.dataset.max_seq_length}) exceeds model "
                    f"max_position_embeddings ({max_pos}). Sequence length is unsupported by base model."
                ],
                "measured_peak_allocated_mb": 0.0,
                "measured_step_delta_mb": 0.0,
                "available_device_memory_mb": 0.0,
                "memory_headroom_percent": 0.0,
                "analytical_estimate": None,
                "error_detail": (
                    f"Configured max_seq_length ({config.dataset.max_seq_length}) exceeds model "
                    f"max_position_embeddings ({max_pos}). Probe rejected."
                ),
                "probe_shape": {"batch_size": batch_size, "seq_len": config.dataset.max_seq_length},
            }

        seq_len = config.dataset.max_seq_length
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", 1000)

        # 2. Orthogonal PEFT / LoRA Adapter Setup
        is_lora_configured = (
            config.task_type == TaskType.LORA
            or getattr(config, "adaptation_mode", None) == "lora"
            or (hasattr(config, "peft") and config.peft is not None)
        )
        if is_lora_configured and not hasattr(model, "peft_config"):
            try:
                from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model

                peft_cfg = getattr(config, "peft", None)
                if peft_cfg is not None:
                    lora_r = getattr(peft_cfg, "r", 16)
                    lora_alpha = getattr(peft_cfg, "lora_alpha", 32)
                    lora_dropout = getattr(peft_cfg, "lora_dropout", 0.05)
                    target_modules = getattr(peft_cfg, "target_modules", None) or ["q_proj", "v_proj"]
                    bias = getattr(peft_cfg, "bias", "none")
                    modules_to_save = getattr(peft_cfg, "modules_to_save", None)
                else:
                    lora_r = 16
                    lora_alpha = 32
                    lora_dropout = 0.05
                    target_modules = ["q_proj", "v_proj"]
                    bias = "none"
                    modules_to_save = None

                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias=bias,
                    modules_to_save=modules_to_save,
                    task_type=PeftTaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)

                trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if trainable_params_count == 0:
                    return {
                        "probe_successful": False,
                        "isolation_mode": "in_process",
                        "verdict": "UNSUPPORTED_CONFIGURATION",
                        "device": str(device),
                        "measurement_type": "none",
                        "gpu_fit_guaranteed": False,
                        "is_representative": False,
                        "error_detail": f"LoRA target_modules {target_modules} did not match model layers (0 trainable parameters).",
                        "measured_peak_allocated_mb": 0.0,
                        "measured_step_delta_mb": 0.0,
                        "available_device_memory_mb": 0.0,
                        "memory_headroom_percent": 0.0,
                        "analytical_estimate": None,
                        "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                    }
            except Exception as peft_err:
                return {
                    "probe_successful": False,
                    "isolation_mode": "in_process",
                    "verdict": "UNSUPPORTED_CONFIGURATION",
                    "device": str(device),
                    "measurement_type": "none",
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "error_detail": f"Failed to instantiate LoRA adapter for configured task: {peft_err}",
                    "measured_peak_allocated_mb": 0.0,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                }

        # 3. Precision & Dtype Setup (matching real training backend / AMP semantics)
        # Upstream training backends (transformers.Trainer / PyTorch AMP) preserve loaded model
        # weights in their declared precision (typically float32) and use torch.autocast for forward operations.
        # Downcasting the entire model to half-precision is avoided to preserve weight fidelity.
        autocast_ctx = contextlib.nullcontext()
        scaler = None

        if config.hardware.mixed_precision == "bf16":
            if device.type == "cuda":
                if not torch.cuda.is_bf16_supported():
                    return {
                        "probe_successful": False,
                        "isolation_mode": "in_process",
                        "verdict": "UNSUPPORTED_HARDWARE_PRECISION",
                        "device": str(device),
                        "measurement_type": "none",
                        "gpu_fit_guaranteed": False,
                        "is_representative": False,
                        "error_detail": "BF16 requested but target CUDA device does not support native BF16.",
                        "measured_peak_allocated_mb": 0.0,
                        "measured_step_delta_mb": 0.0,
                        "available_device_memory_mb": 0.0,
                        "memory_headroom_percent": 0.0,
                        "analytical_estimate": None,
                        "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                    }
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif device.type == "cpu":
                if hasattr(torch, "bfloat16"):
                    autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        elif config.hardware.mixed_precision == "fp16":
            if device.type == "cpu":
                return {
                    "probe_successful": False,
                    "isolation_mode": "in_process",
                    "verdict": "UNSUPPORTED_CONFIGURATION",
                    "device": str(device),
                    "measurement_type": "none",
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "error_detail": "FP16 mixed precision is not supported on CPU execution.",
                    "measured_peak_allocated_mb": 0.0,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                }
            elif device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
                    scaler = torch.cuda.amp.GradScaler()
        elif config.hardware.mixed_precision in ["no", None]:
            pass
        else:
            return {
                "probe_successful": False,
                "isolation_mode": "in_process",
                "verdict": "UNSUPPORTED_CONFIGURATION",
                "device": str(device),
                "measurement_type": "none",
                "gpu_fit_guaranteed": False,
                "is_representative": False,
                "error_detail": f"Unsupported mixed precision mode: '{config.hardware.mixed_precision}'",
                "measured_peak_allocated_mb": 0.0,
                "measured_step_delta_mb": 0.0,
                "available_device_memory_mb": 0.0,
                "memory_headroom_percent": 0.0,
                "analytical_estimate": None,
                "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
            }

        # 4. Gradient Checkpointing Setup
        if config.hardware.gradient_checkpointing:
            try:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception as gc_err:
                return {
                    "probe_successful": False,
                    "isolation_mode": "in_process",
                    "verdict": "UNSUPPORTED_CONFIGURATION",
                    "device": str(device),
                    "measurement_type": "none",
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "error_detail": f"Failed to enable gradient checkpointing: {gc_err}",
                    "measured_peak_allocated_mb": 0.0,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                }

        # 5. DPO Reference Model Preparation (if task is DPO and not using LoRA)
        ref_model = None
        if config.task_type == TaskType.DPO and not is_lora_configured:
            try:
                ref_model = copy.deepcopy(model)
                ref_model.to(device)
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad = False
            except Exception as dpo_ref_err:
                return {
                    "probe_successful": False,
                    "isolation_mode": "in_process",
                    "verdict": "UNSUPPORTED_CONFIGURATION",
                    "device": str(device),
                    "measurement_type": "none",
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "error_detail": f"Full fine-tuning DPO requires frozen reference model in memory; setup failed: {dpo_ref_err}",
                    "measured_peak_allocated_mb": 0.0,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                }

        # Cleanup memory before starting forward-backward measurement
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_mem = torch.cuda.memory_allocated(device)
        else:
            proc = psutil.Process()
            baseline_mem = proc.memory_info().rss

        model.to(device)
        model.train()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        analytical = MemoryEstimator.estimate(config, total_params, trainable_params)

        probe_success = False
        error_msg = None
        peak_allocated = 0
        optimizer = None
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            opt_cfg = getattr(config, "optimizer", None)
            opt_type = getattr(opt_cfg, "optimizer_type", "adamw_torch") if opt_cfg else "adamw_torch"
            lr = getattr(opt_cfg, "learning_rate", 2e-5) if opt_cfg else 2e-5
            wd = getattr(opt_cfg, "weight_decay", 0.01) if opt_cfg else 0.01

            if opt_type in ["adamw_torch", "adamw_hf"]:
                b1 = getattr(opt_cfg, "adam_beta1", 0.9) if opt_cfg else 0.9
                b2 = getattr(opt_cfg, "adam_beta2", 0.999) if opt_cfg else 0.999
                eps = getattr(opt_cfg, "adam_epsilon", 1e-8) if opt_cfg else 1e-8
                optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
            elif opt_type == "sgd":
                optimizer = torch.optim.SGD(trainable, lr=lr, weight_decay=wd)
            elif opt_type == "adamw_torch_fused":
                if device.type != "cuda":
                    return {
                        "probe_successful": False,
                        "isolation_mode": "in_process",
                        "verdict": "UNSUPPORTED_CONFIGURATION",
                        "device": str(device),
                        "measurement_type": "none",
                        "gpu_fit_guaranteed": False,
                        "is_representative": False,
                        "error_detail": "adamw_torch_fused optimizer requires a CUDA device.",
                        "measured_peak_allocated_mb": 0.0,
                        "measured_step_delta_mb": 0.0,
                        "available_device_memory_mb": 0.0,
                        "memory_headroom_percent": 0.0,
                        "analytical_estimate": None,
                        "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                    }
                optimizer = torch.optim.AdamW(trainable, lr=lr, fused=True)
            else:
                return {
                    "probe_successful": False,
                    "isolation_mode": "in_process",
                    "verdict": "UNSUPPORTED_CONFIGURATION",
                    "device": str(device),
                    "measurement_type": "none",
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "error_detail": f"Unsupported optimizer type: '{opt_type}'",
                    "measured_peak_allocated_mb": 0.0,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
                }

        try:
            with autocast_ctx:
                # Objective-specific forward and backward passes
                if config.task_type == TaskType.DPO:
                    # Upstream Core DPO contract:
                    # Generates chosen and rejected sequence pairs
                    dummy_chosen = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    dummy_rejected = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    prompt_len = max(1, seq_len // 2)

                    def _seq_logps(m, input_ids):
                        out = m(input_ids)
                        logits = out.logits if hasattr(out, "logits") else out[0]
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()
                        log_probs = F.log_softmax(shift_logits, dim=-1)
                        per_token = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                        mask = torch.ones_like(shift_labels, dtype=torch.bool)
                        if prompt_len > 1:
                            mask[:, :prompt_len - 1] = False
                        return (per_token * mask).sum(dim=-1)

                    # 1. Policy forward pass (tracks gradients)
                    policy_chosen_logps = _seq_logps(model, dummy_chosen)
                    policy_rejected_logps = _seq_logps(model, dummy_rejected)

                    # 2. Reference forward pass (no gradients)
                    with torch.no_grad():
                        if is_lora_configured and hasattr(model, "disable_adapter"):
                            with model.disable_adapter():
                                ref_chosen_logps = _seq_logps(model, dummy_chosen)
                                ref_rejected_logps = _seq_logps(model, dummy_rejected)
                        elif ref_model is not None:
                            ref_chosen_logps = _seq_logps(ref_model, dummy_chosen)
                            ref_rejected_logps = _seq_logps(ref_model, dummy_rejected)
                        else:
                            raise RuntimeError("DPO reference evaluation failed: missing reference model and adapter disable.")

                    # 3. Preference objective loss calculation (matching Core DPO / TRL DPOTrainer)
                    beta = getattr(config, "dpo_beta", 0.1)
                    pi_logratios = policy_chosen_logps - policy_rejected_logps
                    ref_logratios = ref_chosen_logps - ref_rejected_logps
                    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
                else:
                    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    dummy_labels = dummy_input.clone()
                    outputs = model(input_ids=dummy_input, labels=dummy_labels)
                    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs[0]

            if scaler is not None:
                scaler.scale(loss).backward()
                if optimizer is not None:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                if optimizer is not None:
                    optimizer.step()

            # Record peak memory
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device)
            else:
                peak_allocated = proc.memory_info().rss

            probe_success = True
        except RuntimeError as e:
            error_msg = str(e)
            probe_success = False
        except Exception as e:
            error_msg = str(e)
            probe_success = False
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                del optimizer
            if ref_model is not None:
                del ref_model
            model.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mem_delta_mb = round((peak_allocated - baseline_mem) / (1024 ** 2), 2)
        total_peak_mb = round(peak_allocated / (1024 ** 2), 2)

        # Audit probe representativeness vs requested configuration
        representativeness_warnings: List[str] = []
        is_representative = True

        if device.type != "cuda":
            is_representative = False
            representativeness_warnings.append(
                "Probe executed on CPU host memory. Sampled CPU RSS cannot guarantee discrete GPU VRAM fit or headroom."
            )

        # Hardware safety headroom check
        hardware = HardwareInspector.inspect()
        if device.type == "cuda" and hardware["gpus"]:
            available_mb = hardware["gpus"][0]["total_memory_gb"] * 1024
        else:
            available_mb = hardware["system_ram_available_gb"] * 1024

        headroom_percent = round(((available_mb - total_peak_mb) / available_mb) * 100, 2) if available_mb > 0 else 0.0

        if not probe_success:
            verdict = "OOM_PREVENTED" if "out of memory" in (error_msg or "").lower() else "PROBE_FAILED"
        elif device.type != "cuda" and config.hardware.target_device == "cuda":
            verdict = "PASS_CPU_ONLY_UNVERIFIED_GPU"
        elif headroom_percent < 15.0:
            verdict = "WARNING_HIGH_MEMORY_PRESSURE"
        elif not is_representative:
            verdict = "PASS_NON_REPRESENTATIVE_HEADROOM"
        else:
            verdict = "PASS_SAFE_HEADROOM"

        return {
            "probe_successful": probe_success,
            "isolation_mode": "in_process",
            "verdict": verdict,
            "device": str(device),
            "measurement_type": "cuda_max_memory_allocated" if device.type == "cuda" else "sampled_process_rss",
            "gpu_fit_guaranteed": False,  # Finite representative probe cannot guarantee whole-run fit; operators rely on measured shape and headroom.
            "is_representative": is_representative,
            "representativeness_warnings": representativeness_warnings,
            "measured_peak_allocated_mb": total_peak_mb,
            "measured_step_delta_mb": max(mem_delta_mb, 0.0),
            "available_device_memory_mb": round(available_mb, 2),
            "memory_headroom_percent": headroom_percent,
            "analytical_estimate": analytical,
            "error_detail": error_msg,
            "probe_shape": {"batch_size": batch_size, "seq_len": seq_len},
        }

    @staticmethod
    def run_probe_subprocess(
        model_path: Union[str, Path],
        config: TrainingJobConfig,
        timeout_seconds: float = 60,
        max_process_memory_mb: Optional[int] = 4096,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the probe in an isolated subprocess with ongoing stream draining, aggregate memory tracking and timeout limits."""
        memory_ceiling = max_process_memory_mb if max_process_memory_mb is not None else EmpiricalMemoryProbe.DEFAULT_MEMORY_CEILING_MB

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg_path = tmp_path / "probe_config.json"
            out_path = tmp_path / "probe_output.json"

            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(config.model_dump_json())

            cmd = [
                sys.executable,
                "-m",
                "llm_training_stack.preflight.probe_worker",
                "--config-file",
                str(cfg_path),
                "--model-path",
                str(model_path),
                "--output-file",
                str(out_path),
            ]
            if device:
                cmd.extend(["--device", device])

            # Prepare process environment
            env = os.environ.copy()
            project_root = Path(__file__).resolve().parent.parent.parent
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{str(project_root)}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(project_root)

            start_time = time.time()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                close_fds=True,
                env=env,
            )

            # Continuous stream draining to prevent pipe deadlock
            stdout_drainer = BoundedStreamDrainer(proc.stdout, max_bytes=32768)
            stderr_drainer = BoundedStreamDrainer(proc.stderr, max_bytes=32768)

            timed_out = False
            memory_exceeded = False
            peak_proc_rss_mb = 0.0
            monitor_warnings: List[str] = []

            # Polling loop to enforce aggregate process memory limit and timeout
            try:
                psutil_proc = psutil.Process(proc.pid)
            except Exception:
                psutil_proc = None

            while proc.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    timed_out = True
                    _cleanup_process_tree(proc, psutil_proc)
                    break

                if psutil_proc is not None:
                    try:
                        current_aggregate_rss = _get_aggregate_rss_mb(psutil_proc)
                        peak_proc_rss_mb = max(peak_proc_rss_mb, current_aggregate_rss)
                        if current_aggregate_rss > memory_ceiling:
                            memory_exceeded = True
                            _cleanup_process_tree(proc, psutil_proc)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as proc_err:
                        monitor_warnings.append(f"Process monitor warning: {proc_err}")

                time.sleep(0.05)

            try:
                proc.wait(timeout=3)
            except Exception:
                _cleanup_process_tree(proc, psutil_proc)

            stdout_bounded = stdout_drainer.get_output(timeout_sec=2.0)
            stderr_bounded = stderr_drainer.get_output(timeout_sec=2.0)

            if timed_out:
                return {
                    "probe_successful": False,
                    "isolation_mode": "subprocess",
                    "verdict": "PROBE_TIMEOUT",
                    "device": device or config.hardware.target_device,
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "measured_peak_allocated_mb": peak_proc_rss_mb,
                    "aggregate_process_memory_mb": peak_proc_rss_mb,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "error_detail": f"Subprocess probe timed out after {timeout_seconds} seconds.",
                    "probe_shape": {
                        "batch_size": config.hardware.per_device_train_batch_size,
                        "seq_len": config.dataset.max_seq_length,
                    },
                }

            if memory_exceeded:
                return {
                    "probe_successful": False,
                    "isolation_mode": "subprocess",
                    "verdict": "PROCESS_MEMORY_LIMIT_EXCEEDED",
                    "device": device or config.hardware.target_device,
                    "gpu_fit_guaranteed": False,
                    "is_representative": False,
                    "measured_peak_allocated_mb": peak_proc_rss_mb,
                    "aggregate_process_memory_mb": peak_proc_rss_mb,
                    "measured_step_delta_mb": 0.0,
                    "available_device_memory_mb": 0.0,
                    "memory_headroom_percent": 0.0,
                    "analytical_estimate": None,
                    "error_detail": (
                        f"Subprocess exceeded process memory ceiling ({peak_proc_rss_mb} MB > "
                        f"{memory_ceiling} MB)."
                    ),
                    "probe_shape": {
                        "batch_size": config.hardware.per_device_train_batch_size,
                        "seq_len": config.dataset.max_seq_length,
                    },
                }

            # If output file exists, load result
            if out_path.exists():
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    report["isolation_mode"] = "subprocess"
                    report["aggregate_process_memory_mb"] = peak_proc_rss_mb
                    report["gpu_fit_guaranteed"] = False  # Finite representative probe cannot guarantee whole-run fit; operators rely on measured shape and headroom.
                    return report
                except Exception as parse_err:
                    stderr_bounded += f"\nFailed to parse output JSON: {parse_err}"

            # If failed without output file
            error_msg = stderr_bounded.strip() or stdout_bounded.strip() or "Unknown subprocess failure"
            is_oom = "out of memory" in error_msg.lower() or proc.returncode in [-9, 137]
            verdict = "OOM_PREVENTED" if is_oom else "SUBPROCESS_FAILED"

            return {
                "probe_successful": False,
                "isolation_mode": "subprocess",
                "verdict": verdict,
                "device": device or config.hardware.target_device,
                "gpu_fit_guaranteed": False,
                "is_representative": False,
                "measured_peak_allocated_mb": peak_proc_rss_mb,
                "aggregate_process_memory_mb": peak_proc_rss_mb,
                "measured_step_delta_mb": 0.0,
                "available_device_memory_mb": 0.0,
                "memory_headroom_percent": 0.0,
                "analytical_estimate": None,
                "error_detail": error_msg,
                "probe_shape": {
                    "batch_size": config.hardware.per_device_train_batch_size,
                    "seq_len": config.dataset.max_seq_length,
                },
            }
