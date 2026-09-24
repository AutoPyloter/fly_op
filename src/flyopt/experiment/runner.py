"""Config -> run -> artifact scaffolding (PROTOCOL.md section 5, Faz 0).

Every run records its git commit hash, a content hash of its config, the
seed, and basic environment info, so any number in results/ can be traced
back to exactly the code and settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return out.stdout.strip() + suffix
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _env_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
    }


@dataclass
class RunRecord:
    run_id: str
    config: dict[str, Any]
    config_hash: str
    git_commit: str
    seed: int
    started_at: str
    env: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    finished_at: str | None = None


class ExperimentRunner:
    """Not a job scheduler — just enforces that every run is stamped and
    written to results/ before its metrics can be trusted or compared.
    """

    def __init__(self, repo_dir: Path, results_dir: Path):
        self.repo_dir = Path(repo_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def start(self, config: dict[str, Any], seed: int) -> RunRecord:
        config_hash = _config_hash(config)
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{config_hash}_seed{seed}"
        return RunRecord(
            run_id=run_id,
            config=config,
            config_hash=config_hash,
            git_commit=_git_commit_hash(self.repo_dir),
            seed=seed,
            started_at=datetime.now(timezone.utc).isoformat(),
            env=_env_info(),
        )

    def finish(self, record: RunRecord, metrics: dict[str, Any], positive: bool) -> Path:
        record.metrics = metrics
        record.finished_at = datetime.now(timezone.utc).isoformat()
        subdir = "positive" if positive else "negative"
        out_dir = self.results_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{record.run_id}.json"
        out_path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
        return out_path
