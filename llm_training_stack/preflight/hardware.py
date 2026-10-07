"""Hardware inspection and platform telemetry."""

import os
import platform
import psutil
import torch
from typing import Dict, Any, List


class HardwareInspector:
    """Detects available compute hardware, CPU/GPU details, and memory limits."""

    @staticmethod
    def inspect() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
        cpu_count_logical = psutil.cpu_count(logical=True) or 1

        cuda_available = torch.cuda.is_available()
        gpu_info: List[Dict[str, Any]] = []

        if cuda_available:
            device_count = torch.cuda.device_count()
            for idx in range(device_count):
                props = torch.cuda.get_device_properties(idx)
                gpu_info.append({
                    "device_index": idx,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major_capability": props.major,
                    "minor_capability": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                })
        else:
            device_count = 0

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "cpu_count_physical": cpu_count_physical,
            "cpu_count_logical": cpu_count_logical,
            "system_ram_total_bytes": vm.total,
            "system_ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "system_ram_available_bytes": vm.available,
            "system_ram_available_gb": round(vm.available / (1024 ** 3), 2),
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "gpus": gpu_info,
            "recommended_device": "cuda" if cuda_available else "cpu",
        }
