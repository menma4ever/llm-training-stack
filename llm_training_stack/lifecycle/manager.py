"""Shared Core Durable Job Lifecycle Management."""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from llm_training_stack.config.schema import TaskType, TrainingJobConfig
from llm_training_stack.config.loader import ConfigLoader
from llm_training_stack.pipelines.cpt import CPTPipeline
from llm_training_stack.pipelines.dpo import DPOPipeline
from llm_training_stack.pipelines.lora import LoRAPipeline
from llm_training_stack.pipelines.sft import SFTPipeline

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


REGISTRY_FILE = Path.home() / ".train_stack" / "jobs_registry.json"


def _register_job(job_id: str, run_dir: Union[str, Path]) -> None:
    """Records job ID mapping to run directory in global durable registry with atomic persistence."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved = False
    last_err = None
    for attempt in range(5):
        try:
            reg = {}
            if REGISTRY_FILE.exists():
                try:
                    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                        reg = json.load(f)
                except Exception:
                    reg = {}
            reg[job_id] = str(Path(run_dir).resolve())
            temp_file = REGISTRY_FILE.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
            os.replace(temp_file, REGISTRY_FILE)
            saved = True
            break
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(0.05)
    if not saved and last_err is not None:
        raise OSError(f"Failed to record job {job_id} in global registry after 5 attempts: {last_err}")


def _lookup_job_dir(job_dir_or_id: Union[str, Path]) -> Path:
    """Resolves job directory from path, job ID, or global registry."""
    p = Path(job_dir_or_id)
    if (p / "job_state.json").exists():
        return p
    if p.is_file() and p.name == "job_state.json":
        return p.parent

    # If p is a parent output directory containing job subdirectories
    if p.is_dir():
        job_states = list(p.glob("*/job_state.json"))
        if job_states:
            job_states.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return job_states[0].parent

    job_str = str(job_dir_or_id).strip()
    # Check global registry
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                reg = json.load(f)
            if job_str in reg:
                candidate = Path(reg[job_str])
                if (candidate / "job_state.json").exists():
                    return candidate
        except Exception:
            pass

    # Check common output roots
    for root in [Path("."), Path("./runs"), Path("./jobs")]:
        candidate = root / job_str
        if (candidate / "job_state.json").exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Job not found for ID or directory: {job_dir_or_id}")


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.SUBMITTED
    task_type: TaskType
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_dir: str
    config: Dict[str, Any]
    current_step: int = 0
    total_steps: int = 0
    latest_loss: Optional[float] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    checkpoint_dir: Optional[str] = None

    def save(self, target_dir: Union[str, Path]) -> None:
        p = Path(target_dir) / "job_state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        saved = False
        last_err = None
        for attempt in range(5):
            try:
                temp_p = p.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}")
                with open(temp_p, "w", encoding="utf-8") as f:
                    json.dump(self.model_dump(mode="json"), f, indent=2)
                os.replace(temp_p, p)
                saved = True
                break
            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(0.05)
        if not saved and last_err is not None:
            raise OSError(f"Failed to persist job state to {p} after 5 attempts: {last_err}")

    @classmethod
    def load(cls, target_dir: Union[str, Path]) -> "JobRecord":
        p = _lookup_job_dir(target_dir)
        state_file = p / "job_state.json" if p.is_dir() else p
        if not state_file.exists():
            raise FileNotFoundError(f"Job state not found at {state_file}")
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class LifecycleManager:
    """Manages asynchronous and durable training job lifecycles.
    Provides submission, live status monitoring, structured log tailing, cancellation, and resumption.
    """

    _active_threads: Dict[str, threading.Thread] = {}
    _cancel_flags: Dict[str, threading.Event] = {}

    @classmethod
    def resolve_job_dir(cls, job_dir_or_id: Union[str, Path]) -> Path:
        return _lookup_job_dir(job_dir_or_id)

    @classmethod
    def create_pipeline(cls, config: TrainingJobConfig):
        if config.task_type == TaskType.CPT:
            return CPTPipeline(config)
        elif config.task_type == TaskType.SFT:
            return SFTPipeline(config)
        elif config.task_type == TaskType.DPO:
            return DPOPipeline(config)
        elif config.task_type == TaskType.LORA:
            return LoRAPipeline(config)
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")

    @classmethod
    def run_worker(cls, run_dir: Union[str, Path]) -> JobRecord:
        """Executes a training job from its directory and updates durable lifecycle records."""
        resolved_dir = cls.resolve_job_dir(run_dir)
        record = JobRecord.load(resolved_dir)
        config = ConfigLoader.load_from_dict(record.config)

        cancel_token = resolved_dir / "cancel.token"

        def _is_cancelled() -> bool:
            return cancel_token.exists()

        if _is_cancelled():
            record.status = JobStatus.CANCELLED
            record.finished_at = datetime.now(timezone.utc).isoformat()
            record.error = "Cancelled by operator request."
            record.save(resolved_dir)
            return record

        record.status = JobStatus.RUNNING
        record.started_at = datetime.now(timezone.utc).isoformat()
        record.save(resolved_dir)

        try:
            pipeline = cls.create_pipeline(config)
            resume_from = record.checkpoint_dir
            res = pipeline.train(resume_from=resume_from, cancel_check=_is_cancelled)

            if isinstance(res, dict):
                record.metrics = res
                if "final_loss" in res and res["final_loss"] is not None:
                    record.latest_loss = float(res["final_loss"])
                if "total_steps" in res and res["total_steps"] is not None:
                    record.current_step = int(res["total_steps"])
                if "final_checkpoint" in res and res["final_checkpoint"] is not None:
                    record.checkpoint_dir = str(res["final_checkpoint"])

            if _is_cancelled() or (isinstance(res, dict) and res.get("status") == "CANCELLED"):
                record.status = JobStatus.CANCELLED
                record.error = "Cancelled by operator request."
            else:
                record.status = JobStatus.COMPLETED
            record.finished_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            if _is_cancelled():
                record.status = JobStatus.CANCELLED
                record.error = "Job cancelled by operator request"
            else:
                record.status = JobStatus.FAILED
                record.error = str(exc)
            record.finished_at = datetime.now(timezone.utc).isoformat()
            logger.exception(f"Job {record.job_id} encountered execution error: {exc}")
        finally:
            try:
                events = cls.get_job_logs(resolved_dir, tail_lines=5)
                if events:
                    last_ev = events[-1]
                    if "step" in last_ev and (record.current_step is None or record.current_step == 0):
                        record.current_step = int(last_ev["step"])
                    if "loss" in last_ev and record.latest_loss is None:
                        record.latest_loss = float(last_ev["loss"])
                if not record.checkpoint_dir:
                    ckpts = sorted(list(resolved_dir.glob("checkpoint-*")), key=lambda p: p.stat().st_mtime)
                    if ckpts:
                        record.checkpoint_dir = str(ckpts[-1])
            except Exception:
                pass
            record.save(resolved_dir)

        return record

    @classmethod
    def submit_job(
        cls,
        config: TrainingJobConfig,
        resume_from: Optional[str] = None,
        run_in_background: bool = False,
        isolate_job_dir: bool = False,
    ) -> JobRecord:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        base_dir = Path(config.logging.output_dir).resolve()

        # Isolate job run directory to subfolder only when requested
        if isolate_job_dir:
            output_dir = base_dir / job_id
        else:
            output_dir = base_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config.logging.output_dir = str(output_dir)

        record = JobRecord(
            job_id=job_id,
            status=JobStatus.SUBMITTED,
            task_type=config.task_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            run_dir=str(output_dir),
            config=config.model_dump(mode="json"),
            total_steps=getattr(config, "max_steps", 100) or 100,
            checkpoint_dir=str(resume_from) if resume_from else None,
        )
        record.save(output_dir)
        _register_job(job_id, output_dir)

        if run_in_background:
            cmd = [sys.executable, "-m", "llm_training_stack.cli.main", "run-worker", "--run-dir", str(output_dir)]
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                **kwargs,
            )
        else:
            cls.run_worker(output_dir)

        return JobRecord.load(output_dir)

    @classmethod
    def get_job_status(cls, job_dir_or_id: Union[str, Path]) -> JobRecord:
        """Inspects durable job state on disk by directory or resolved job ID."""
        run_dir = cls.resolve_job_dir(job_dir_or_id)
        return JobRecord.load(run_dir)

    @classmethod
    def get_job_logs(cls, job_dir_or_id: Union[str, Path], tail_lines: int = 50) -> List[Dict[str, Any]]:
        """Reads recent structured events from the job's events.jsonl log."""
        run_dir = cls.resolve_job_dir(job_dir_or_id)
        events_file = run_dir / "events.jsonl" if run_dir.is_dir() else run_dir
        if not events_file.exists():
            return []

        lines = []
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)

        recent = lines[-tail_lines:] if len(lines) > tail_lines else lines
        events = []
        for l in recent:
            try:
                events.append(json.loads(l))
            except Exception:
                events.append({"raw": l.strip()})
        return events

    @classmethod
    def cancel_job(cls, job_dir_or_id: Union[str, Path], wait: bool = True, timeout: float = 10.0) -> JobRecord:
        """Cancels an active training job across processes and updates durable state."""
        run_dir = cls.resolve_job_dir(job_dir_or_id)
        record = cls.get_job_status(run_dir)

        # 1. Signal cancellation cross-process via token file
        cancel_token = Path(run_dir) / "cancel.token"
        cancel_token.write_text(f"cancelled_at={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")

        # 2. Signal in-memory thread event
        if record.job_id in cls._cancel_flags:
            cls._cancel_flags[record.job_id].set()

        # 3. For SUBMITTED jobs (worker hasn't started yet), mark CANCELLED immediately
        if record.status == JobStatus.SUBMITTED:
            record.status = JobStatus.CANCELLED
            record.finished_at = datetime.now(timezone.utc).isoformat()
            record.error = "Cancelled by operator request."
            record.save(run_dir)
            return record

        # 4. If wait is requested and job is RUNNING, poll for child acknowledgment
        if wait and record.status == JobStatus.RUNNING:
            deadline = time.time() + timeout
            while time.time() < deadline:
                current = cls.get_job_status(run_dir)
                if current.status == JobStatus.CANCELLED and current.finished_at is not None:
                    return current
                time.sleep(0.1)
        # 5. Return true disk status without premature terminal cancellation:
        # If child worker has not acknowledged or timeout expired, status remains RUNNING and finished_at is unset.
        return cls.get_job_status(run_dir)

