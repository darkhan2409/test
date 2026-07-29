"""Quarantine — §34.

Правило регламента жёсткое: «запрещено молча удалять проблемные записи без
метрики и lineage». Поэтому здесь метрика не опциональна — `Quarantine`
держит ссылку на монитор и инкрементит счётчик сам. Отправить запись в
карантин и забыть про метрику технически невозможно.

`processing_time` берётся не из `datetime.now()` внутри, а передаётся один раз
на прогон. Иначе файлы карантина отличались бы между прогонами, и сравнение
single-worker vs multi-worker (§29 п.10) пришлось бы делать «кроме карантина».
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .monitor import DataQualityMonitor, Metric, Total

UTC = timezone.utc


class ReasonCode(StrEnum):
    """Причины карантина — ровно перечень §34."""

    UNRESOLVED_CLIENT_ID = "unresolved_client_id"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    MISSING_EVENT_TIME = "missing_event_time"
    UNKNOWN_TIMEZONE = "unknown_timezone"
    SOURCE_CONTRACT_VIOLATION = "source_contract_violation"
    UNRESOLVED_DUPLICATE_CONFLICT = "unresolved_duplicate_conflict"
    INCOMPATIBLE_SCHEMA_VERSION = "incompatible_schema_version"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"


# Какую метрику §33 поднимает каждая причина. Две причины своей метрики
# в §33 не имеют — identity resolution и несовпадение content hash там просто
# не описаны; они считаются только общим счётчиком карантина.
REASON_METRIC: dict[ReasonCode, Metric | None] = {
    ReasonCode.UNRESOLVED_CLIENT_ID: None,
    ReasonCode.UNKNOWN_EVENT_TYPE: Metric.UNKNOWN_EVENT_TYPE_RATE,
    ReasonCode.MISSING_EVENT_TIME: Metric.TIMESTAMP_ERROR_RATE,
    ReasonCode.UNKNOWN_TIMEZONE: Metric.TIMESTAMP_ERROR_RATE,
    ReasonCode.SOURCE_CONTRACT_VIOLATION: Metric.SCHEMA_VIOLATION_RATE,
    ReasonCode.UNRESOLVED_DUPLICATE_CONFLICT: Metric.DEDUP_CONFLICT_RATE,
    ReasonCode.INCOMPATIBLE_SCHEMA_VERSION: Metric.SCHEMA_VIOLATION_RATE,
    ReasonCode.CONTENT_HASH_MISMATCH: None,
}

# Можно ли починить запись, не меняя её саму. Неизвестный тип события чинится
# обновлением маппинга, несовпадение hash — нет.
REASON_RECOVERABLE: dict[ReasonCode, bool] = {
    ReasonCode.UNRESOLVED_CLIENT_ID: True,
    ReasonCode.UNKNOWN_EVENT_TYPE: True,
    ReasonCode.MISSING_EVENT_TIME: False,
    ReasonCode.UNKNOWN_TIMEZONE: True,
    ReasonCode.SOURCE_CONTRACT_VIOLATION: True,
    ReasonCode.UNRESOLVED_DUPLICATE_CONFLICT: True,
    ReasonCode.INCOMPATIBLE_SCHEMA_VERSION: True,
    ReasonCode.CONTENT_HASH_MISMATCH: False,
}


@dataclass(frozen=True)
class QuarantineEntry:
    """Запись карантина — обязательные поля §34."""

    source: str
    raw_reference: str
    reason_code: str
    recoverable: bool
    processing_time: str
    pipeline_version: str
    detail: str
    partition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "raw_reference": self.raw_reference,
            "partition": self.partition,
            "reason_code": self.reason_code,
            "recoverable": self.recoverable,
            "processing_time": self.processing_time,
            "pipeline_version": self.pipeline_version,
            "detail": self.detail,
        }


class Quarantine:
    """Накопитель карантина с обязательной привязкой к метрикам."""

    def __init__(
        self,
        monitor: DataQualityMonitor,
        *,
        processing_time: datetime,
        pipeline_version: str,
    ) -> None:
        if processing_time.tzinfo is None or processing_time.utcoffset() is None:
            raise ValueError("processing_time должен быть с часовым поясом")
        self._monitor = monitor
        self._processing_time = processing_time.astimezone(UTC).isoformat().replace("+00:00", "Z")
        self._pipeline_version = pipeline_version
        self._entries: list[QuarantineEntry] = []

    def add(
        self,
        reason: ReasonCode,
        *,
        source: str,
        raw_reference: str,
        detail: str = "",
        partition: str = "",
        recoverable: bool | None = None,
        count_metric: bool = True,
    ) -> None:
        """Отправить запись в карантин и одновременно поднять метрику.

        `raw_reference` — то, по чему запись находится в сыром источнике
        (`source_record_id` или путь+смещение): без него нет lineage.

        `count_metric=False` — для случаев, когда метрика считается не по
        записям, а по группам: конфликтующий дубль (§9.3) — свойство группы,
        и группа уже посчитана вызывающим кодом. Умолчание оставлено `True`,
        поэтому забыть метрику по-прежнему невозможно — от неё можно только
        отказаться явно.
        """
        entry = QuarantineEntry(
            source=source,
            raw_reference=raw_reference,
            reason_code=str(reason),
            recoverable=REASON_RECOVERABLE[reason] if recoverable is None else recoverable,
            processing_time=self._processing_time,
            pipeline_version=self._pipeline_version,
            detail=detail,
            partition=partition,
        )
        self._entries.append(entry)

        self._monitor.add_total(Total.QUARANTINED)
        metric = REASON_METRIC[reason]
        if metric is not None and count_metric:
            self._monitor.count(metric)

    def merge(self, other: "Quarantine") -> None:
        self._entries.extend(other._entries)

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.reason_code] = counts.get(entry.reason_code, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        """Блок карантина для отчёта прогона."""
        return {"total": len(self._entries), "by_reason": self.counts_by_reason()}

    def write(self, quarantine_dir: Path) -> list[Path]:
        """Разложить карантин по источникам: `data/quarantine/<source>.jsonl`.

        Порядок строк детерминирован сортировкой, а не порядком, в котором
        воркеры прислали записи.
        """
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(quarantine_dir.glob("*.jsonl")):
            stale.unlink()

        by_source: dict[str, list[QuarantineEntry]] = {}
        for entry in self._entries:
            by_source.setdefault(entry.source, []).append(entry)

        written: list[Path] = []
        for source in sorted(by_source):
            rows = sorted(
                by_source[source],
                key=lambda item: (item.reason_code, item.raw_reference),
            )
            payload = "".join(
                json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            )
            path = quarantine_dir / f"{source}.jsonl"
            path.write_bytes(payload.encode("utf-8"))
            written.append(path)
        return written
