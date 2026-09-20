import json
import hashlib
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .config import ExperimentConfig, save_resolved_config


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
        return None
    return value


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunArtifacts:
    """Persist reproducibility metadata and structured metrics for one run."""

    def __init__(self, save_dir: str, args: Any, experiment: ExperimentConfig):
        self._write_manifest()

    def register(self, key: str, relative_path: str) -> None:
        self.manifest["artifacts"][key] = relative_path
        self._write_manifest()

    def record_inputs(self, paths: Dict[str, str]) -> None:
        self.manifest["inputs"] = {
            name: {
                "path": str(Path(path).resolve()),
                "size_bytes": Path(path).stat().st_size,
                "sha256": _sha256(Path(path)),
            }
            for name, path in paths.items()
        }
        self._write_manifest()

    def append_metrics(self, record: Dict[str, Any]) -> None:
        payload = _serializable(record)
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def complete(self, summary: Dict[str, Any]) -> None:
        self.manifest["status"] = "completed"
        self.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["summary"] = _serializable(summary)
        self._write_manifest()

    def _write_manifest(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as manifest_file:
            yaml.safe_dump(
                _serializable(self.manifest),
                manifest_file,
                allow_unicode=True,
                sort_keys=False,
            )
