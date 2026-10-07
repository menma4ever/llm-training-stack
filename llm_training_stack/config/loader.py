"""Configuration loader, dumper, and validator supporting YAML and JSON."""

import json
from pathlib import Path
from typing import Union, Dict, Any
import yaml
from pydantic import ValidationError

from llm_training_stack.config.schema import TrainingJobConfig


class ConfigLoader:
    """Safely loads, validates, and serializes TrainingJobConfig instances."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> TrainingJobConfig:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        elif path.suffix.lower() == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix}. Use .yaml, .yml or .json.")

        if not isinstance(data, dict):
            raise ValueError(f"Root configuration must be a mapping/dictionary, got {type(data).__name__}")

        return TrainingJobConfig.model_validate(data)

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> TrainingJobConfig:
        return TrainingJobConfig.model_validate(data)

    @staticmethod
    def save_to_file(config: TrainingJobConfig, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="json")
        
        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return path
