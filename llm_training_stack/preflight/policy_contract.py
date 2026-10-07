"""Policy-facing API contract for preflight verification, sandboxing, and resource validation.

Provides a clean interface for Manager, CLI, and MCP servers to validate resource identities,
enforce memory budgets, and run isolated inspections without direct coupling to low-level subprocess plumbing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llm_training_stack.config.schema import TrainingJobConfig
from llm_training_stack.preflight.hardware import HardwareInspector
from llm_training_stack.preflight.memory_model import MemoryEstimator
from llm_training_stack.preflight.resource_identity import (
    ResourceClassifier,
    ResourceIdentity,
    ResourceKind,
)


class PreflightPolicyContract:
    """Manager-facing preflight contract providing typed validation and isolated operations."""

    DEFAULT_PROBE_MEMORY_CEILING_MB = 4096
    DEFAULT_INSPECTION_MEMORY_CEILING_MB = 2048
    DEFAULT_PROBE_TIMEOUT_SECONDS = 60
    DEFAULT_INSPECTION_TIMEOUT_SECONDS = 30

    # Operator hard ceilings & positive bounds (clamps untrusted/arbitrary caller limits)
    OPERATOR_MAX_PROBE_MEMORY_MB = 8192
    OPERATOR_MIN_PROBE_MEMORY_MB = 1
    OPERATOR_MAX_PROBE_TIMEOUT_SECONDS = 300
    OPERATOR_MIN_PROBE_TIMEOUT_SECONDS = 0.001

    OPERATOR_MAX_INSPECTION_MEMORY_MB = 4096
    OPERATOR_MIN_INSPECTION_MEMORY_MB = 1
    OPERATOR_MAX_INSPECTION_TIMEOUT_SECONDS = 120
    OPERATOR_MIN_INSPECTION_TIMEOUT_SECONDS = 0.001

    @classmethod
    def clamp_budget(
        cls,
        requested: Optional[Union[int, float]],
        default_val: float,
        min_val: float,
        max_val: float,
        param_name: str,
    ) -> float:
        """Validates strictly positive bounds and clamps caller request against operator maximum."""
        if requested is None:
            return default_val
        if requested <= 0:
            raise ValueError(f"Operator policy violation: '{param_name}' must be strictly positive (> 0), got {requested}")
        if requested < min_val:
            return min_val
        if requested > max_val:
            return max_val
        return requested

    @classmethod
    def classify_resource(
        cls,
        identifier: Optional[str],
        allow_nonexistent_local: bool = False,
    ) -> ResourceIdentity:
        """Classifies a model, tokenizer, or dataset string into a typed ResourceIdentity."""
        return ResourceClassifier.classify(
            identifier=identifier,
            allow_nonexistent_local=allow_nonexistent_local,
        )

    @classmethod
    def classify_output_resource(cls, identifier: Optional[str]) -> ResourceIdentity:
        """Classifies an output directory target. Output targets are ALWAYS local filesystem paths."""
        return ResourceClassifier.classify_output_path(identifier)

    @classmethod
    def validate_resource_in_roots(
        cls,
        identifier: Optional[str],
        allowed_roots: List[Path],
        field_name: str = "path",
        allow_hub_ids: bool = True,
        is_output_path: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Validates that a path is within permitted roots, or is an allowed Hub ID."""
        identity = (
            cls.classify_output_resource(identifier)
            if is_output_path
            else cls.classify_resource(identifier)
        )

        if identity.kind == ResourceKind.INVALID:
            return False, f"Invalid {field_name}: {identity.validation_error}"

        if identity.is_remote:
            return False, f"Remote URI protocol in {field_name} is forbidden: '{identifier}'"

        if identity.is_hub:
            if not allow_hub_ids:
                return False, f"{field_name} does not permit Hub identifiers: '{identifier}'"
            return True, None

        # Local path check
        is_valid, err_msg = ResourceClassifier.validate_in_roots(identity, allowed_roots)
        if not is_valid:
            return False, f"Path root violation in {field_name}: {err_msg}"

        return True, None

    @classmethod
    def inspect_hardware(cls) -> Dict[str, Any]:
        """Returns structured hardware specs and platform telemetry."""
        return HardwareInspector.inspect()

    @classmethod
    def estimate_memory(
        cls,
        config: TrainingJobConfig,
        total_params: int,
        trainable_params: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analytical static memory calculation with explicit assumptions."""
        return MemoryEstimator.estimate(config, total_params, trainable_params)

    @classmethod
    def run_isolated_probe(
        cls,
        model_path: str,
        config: TrainingJobConfig,
        max_process_memory_mb: Optional[int] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes a bounded empirical probe inside an isolated subprocess with strict budget limits."""
        from llm_training_stack.preflight.probe import EmpiricalMemoryProbe

        memory_ceiling = int(cls.clamp_budget(
            max_process_memory_mb,
            cls.DEFAULT_PROBE_MEMORY_CEILING_MB,
            cls.OPERATOR_MIN_PROBE_MEMORY_MB,
            cls.OPERATOR_MAX_PROBE_MEMORY_MB,
            "max_process_memory_mb",
        ))
        timeout = cls.clamp_budget(
            timeout_seconds,
            cls.DEFAULT_PROBE_TIMEOUT_SECONDS,
            cls.OPERATOR_MIN_PROBE_TIMEOUT_SECONDS,
            cls.OPERATOR_MAX_PROBE_TIMEOUT_SECONDS,
            "timeout_seconds",
        )

        return EmpiricalMemoryProbe.run_probe_subprocess(
            model_path=model_path,
            config=config,
            timeout_seconds=timeout,
            max_process_memory_mb=memory_ceiling,
            device=device,
        )

    @classmethod
    def audit_tokenizer_isolated(
        cls,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        max_process_memory_mb: Optional[int] = None,
        timeout_seconds: Optional[Union[int, float]] = None,
    ) -> Dict[str, Any]:
        """Runs tokenizer and vocabulary preflight inspection in an isolated bounded subprocess."""
        from llm_training_stack.preflight.data_inspector import PreflightInspectionSuite

        memory_ceiling = int(cls.clamp_budget(
            max_process_memory_mb,
            cls.DEFAULT_INSPECTION_MEMORY_CEILING_MB,
            cls.OPERATOR_MIN_INSPECTION_MEMORY_MB,
            cls.OPERATOR_MAX_INSPECTION_MEMORY_MB,
            "max_process_memory_mb",
        ))
        timeout = cls.clamp_budget(
            timeout_seconds,
            cls.DEFAULT_INSPECTION_TIMEOUT_SECONDS,
            cls.OPERATOR_MIN_INSPECTION_TIMEOUT_SECONDS,
            cls.OPERATOR_MAX_INSPECTION_TIMEOUT_SECONDS,
            "timeout_seconds",
        )

        return PreflightInspectionSuite.run_isolated_inspection(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_process_memory_mb=memory_ceiling,
            timeout_seconds=timeout,
        )
