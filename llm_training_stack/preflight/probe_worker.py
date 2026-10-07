"""Worker script executed in an isolated subprocess for bounded memory probing.

This script runs in a separate OS process so that hard CUDA OOM crashes,
allocator faults, or segmentation faults are safely isolated from the host server.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

from llm_training_stack.config.loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description="Isolated Subprocess Memory Probe Worker")
    parser.add_argument("--config-file", required=True, help="Path to JSON config file")
    parser.add_argument("--model-path", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--output-file", required=True, help="Path to write JSON probe report")
    parser.add_argument("--device", default=None, help="Target device (cpu or cuda)")

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config_file, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        config = ConfigLoader.load_from_dict(raw_cfg)

        target_device = None
        if args.device:
            target_device = torch.device(args.device)

        # Import probe here to avoid circular dependencies
        from llm_training_stack.preflight.probe import EmpiricalMemoryProbe

        # Load model preserving configured model dtype (matches training pipeline loading)
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "auto": None,
        }
        model_torch_dtype = getattr(config.model, "torch_dtype", "float32")
        target_dtype = dtype_map.get(model_torch_dtype, None)

        # Load model with trust_remote_code and configured model dtype
        trust_remote_code = getattr(config.model, "trust_remote_code", False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=target_dtype,
        )

        # Execute bounded forward-backward probe
        report = EmpiricalMemoryProbe.run_probe(
            model=model,
            config=config,
            device=target_device,
            in_subprocess=False,  # Inside worker, run directly
        )

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        sys.exit(0)

    except Exception as exc:
        err_report = {
            "probe_successful": False,
            "isolation_mode": "subprocess",
            "verdict": "OOM_PREVENTED" if "out of memory" in str(exc).lower() else "SUBPROCESS_EXCEPTION",
            "gpu_fit_guaranteed": False,
            "is_representative": False,
            "measured_peak_allocated_mb": 0.0,
            "measured_step_delta_mb": 0.0,
            "available_device_memory_mb": 0.0,
            "memory_headroom_percent": 0.0,
            "analytical_estimate": None,
            "error_detail": str(exc),
        }
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(err_report, f, indent=2)
        except Exception:
            pass
        sys.stderr.write(f"PROBE_WORKER_ERROR: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
