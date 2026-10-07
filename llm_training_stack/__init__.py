"""LLM Training Stack — Production Open-Source LLM Training & Fine-Tuning Framework."""

__version__ = "0.1.0"
__author__ = "Abdulaziz Komilov"
__license__ = "Apache-2.0"

from llm_training_stack.config.schema import (
    TaskType,
    TrainingJobConfig,
    ModelConfig,
    DatasetConfig,
    PeftConfig,
    OptimizerConfig,
    HardwareConfig,
    LoggingConfig,
)
from llm_training_stack.preflight.hardware import HardwareInspector
from llm_training_stack.preflight.memory_model import MemoryEstimator
from llm_training_stack.preflight.probe import EmpiricalMemoryProbe
from llm_training_stack.provenance.manifest import RunManifest
from llm_training_stack.provenance.event_logger import StructuredEventLogger
from llm_training_stack.checkpoints.manager import CheckpointManager
from llm_training_stack.checkpoints.resume import ResumeManager
from llm_training_stack.eval.evaluator import Evaluator
from llm_training_stack.eval.comparator import RunComparator

__all__ = [
    "TaskType",
    "TrainingJobConfig",
    "ModelConfig",
    "DatasetConfig",
    "PeftConfig",
    "OptimizerConfig",
    "HardwareConfig",
    "LoggingConfig",
    "HardwareInspector",
    "MemoryEstimator",
    "EmpiricalMemoryProbe",
    "RunManifest",
    "StructuredEventLogger",
    "CheckpointManager",
    "ResumeManager",
    "Evaluator",
    "RunComparator",
]
