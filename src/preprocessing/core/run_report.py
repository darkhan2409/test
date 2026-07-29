"""Отчёт прогона — сводка метрик §33 и карантина §34.

Отчёт диагностический: он не входит ни в `preprocessing_state_sha256`, ни в
обязательные артефакты §31. Пишется человекочитаемо (с отступами), а не
канонически — сравнивать его побайтово никто не будет.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .monitor import DataQualityMonitor
from .quarantine import Quarantine

UTC = timezone.utc

REPORT_NAME = "run_report.json"


def build_run_report(
    *,
    monitor: DataQualityMonitor,
    quarantine: Quarantine,
    run_started_at: datetime,
    versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать отчёт прогона."""
    report: dict[str, Any] = {
        "run_started_at": run_started_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "quarantine": quarantine.summary(),
    }
    report.update(monitor.report())
    if versions is not None:
        report["versions"] = versions
    return report


def write_run_report(prepared_dir: Path, report: dict[str, Any]) -> Path:
    """Записать `data/prepared/run_report.json`."""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    path = prepared_dir / REPORT_NAME
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path
