"""Immutable run manifest recording exact provenance and environmental state."""

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import transformers
import peft
import trl
import accelerate

from llm_training_stack.config.schema import TrainingJobConfig
from llm_training_stack.preflight.hardware import HardwareInspector


class RunManifest:
    """Creates, saves, and validates immutable run manifests."""

    @staticmethod
    def get_git_info() -> Dict[str, Any]:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2
            ).decode("utf-8").strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2
            ).decode("utf-8").strip()
            is_dirty = len(status) > 0
            return {"commit_hash": commit, "is_dirty": is_dirty}
        except Exception:
            return {"commit_hash": "unversioned_or_git_unavailable", "is_dirty": False}

    @classmethod
    def create(cls, config: TrainingJobConfig) -> Dict[str, Any]:
        import hashlib
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        git_info = cls.get_git_info()
        hardware = HardwareInspector.inspect()

        library_versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "accelerate": accelerate.__version__,
        }

        cfg_dump = config.model_dump(mode="json")
        canonical_str = json.dumps(
            {"config": cfg_dump, "git": git_info, "hardware": hardware, "libraries": library_versions},
            sort_keys=True,
        )
        launch_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        manifest = {
            "manifest_version": "1.0.0",
            "run_id": run_id,
            "created_at_utc": now,
            "status": "INITIALIZED",
            "task_type": config.task_type.value,
            "model_name_or_path": config.model.model_name_or_path,
            "model_revision": config.model.revision,
            "config": cfg_dump,
            "libraries": library_versions,
            "git": git_info,
            "hardware": hardware,
            "launch_digest_sha256": launch_digest,
            "completed_at_utc": None,
            "final_metrics": {},
        }
        return manifest

    @classmethod
    def save_launch_manifest(cls, manifest: Dict[str, Any], output_dir: Path) -> Path:
        """Guards launch manifest against post-hoc mutation via write-once contract."""
        output_dir.mkdir(parents=True, exist_ok=True)
        launch_path = output_dir / "launch_manifest.json"
        if launch_path.exists():
            raise RuntimeError("Launch manifest is immutable and cannot be overwritten post-launch.")

        with open(launch_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return launch_path

    @classmethod
    def verify_launch_manifest(cls, output_dir: Path) -> bool:
        """Verifies that launch manifest exists and has not been tampered with post-launch."""
        import hashlib
        launch_path = output_dir / "launch_manifest.json"
        if not launch_path.exists():
            return False
        with open(launch_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stored_digest = data.get("launch_digest_sha256")
        if not stored_digest:
            return False
        canonical_str = json.dumps(
            {
                "config": data.get("config"),
                "git": data.get("git"),
                "hardware": data.get("hardware"),
                "libraries": data.get("libraries"),
            },
            sort_keys=True,
        )
        computed_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return stored_digest == computed_digest

    @classmethod
    def save(cls, manifest: Dict[str, Any], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure write-once launch manifest is initialized on first save
        launch_path = output_dir / "launch_manifest.json"
        if not launch_path.exists():
            cls.save_launch_manifest(manifest, output_dir)

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    @classmethod
    def load(cls, manifest_path: Path) -> Dict[str, Any]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

