"""Structured JSON Lines event logger for real-time telemetry."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredEventLogger:
    """Writes real-time step and lifecycle events to an append-only JSONL log."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a", encoding="utf-8")

    def log_event(self, event_type: str, step: int, data: Dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "step": step,
            **data,
        }
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        epoch: float,
        step_time_ms: float,
        tokens_per_sec: float,
        grad_norm: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> None:
        metrics = {
            "loss": round(loss, 5),
            "learning_rate": lr,
            "epoch": round(epoch, 3),
            "step_time_ms": round(step_time_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = round(grad_norm, 4)
        if memory_mb is not None:
            metrics["memory_mb"] = round(memory_mb, 2)

        self.log_event("step_metric", step, metrics)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
