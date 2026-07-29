"""Запись, проходящая цепочку §37.2.

Компоненты цепочки не переписывают запись целиком, а достраивают её: сначала
известна только сырая строка источника, потом появляется `client_id` (§7),
потом `timestamp_utc` и `calendar_timezone` (§12). Поэтому типы наследуются
друг от друга, а не вкладываются один в другой: доступ остаётся плоским
(`record.payload`, `record.client_id`, `record.timestamp_utc`), и через
пятнадцать компонентов не превращается в `record.record.record.payload`.

Все типы неизменяемые. Новая стадия делает `dataclasses.replace` или собирает
следующий тип — но не правит запись на месте: скрытая мутация в одном
компоненте была бы невидима в дампах соседних (§1.6) и ломала бы разбор.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping


class QualityFlag(StrEnum):
    """Пометки качества записи (§5 `quality_flags`).

    Флаг — не причина отбраковки: запись с флагом продолжает путь. Он
    объясняет, почему значение выглядит странно, и переживает до §32.2.
    """

    FUTURE_TIMESTAMP = "future_timestamp"
    """Событие позже допустимого горизонта (§12 п.9)."""

    LATE_ARRIVING = "late_arriving"
    """Запись пришла позже допустимой задержки контракта (§33.12)."""

    TIMEZONE_FALLBACK = "timezone_fallback"
    """Зона взята из source default, а не из данных (§12 п.7, §33.11)."""

    PROFILE_SNAPSHOT_MISSING = "profile_snapshot_missing"
    """У клиента нет ни одного снимка профиля до T (§6 п.3, §33.14)."""


@dataclass(frozen=True)
class SourceRecord:
    """Сырая запись, прошедшая Source Contract (§4). Выход `SourceReader`.

    `payload` намеренно остаётся сырым: следующие компоненты работают с ним
    по своим правилам. Здесь он лишь признан соответствующим схеме.
    """

    source: str
    partition: str
    line_number: int
    source_record_id: str
    source_schema_version: str
    client_ref: str | None
    payload: Mapping[str, Any]

    @property
    def raw_reference(self) -> str:
        """Ссылка на запись в источнике для lineage и карантина (§8, §34)."""
        return f"{self.partition}#{self.line_number}:{self.source_record_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "partition": self.partition,
            "line_number": self.line_number,
            "source_record_id": self.source_record_id,
            "source_schema_version": self.source_schema_version,
            "client_ref": self.client_ref,
            "payload": dict(self.payload),
        }

    def debug_row(self) -> dict[str, Any]:
        """Строка трассировки (§1.6).

        Сырой объект источника лежит под ключом `record`, а не `payload`:
        `payload` — имя конверта самого дампа, и вложенный `payload.payload`
        читался бы как ошибка.
        """
        return {
            "client_ref": self.client_ref,
            "source_record_id": self.source_record_id,
            "source": self.source,
            "partition": self.partition,
            "line_number": self.line_number,
            "source_schema_version": self.source_schema_version,
            "record": dict(self.payload),
        }


@dataclass(frozen=True)
class IdentifiedRecord(SourceRecord):
    """Запись с разрешённым каноническим `client_id` (§7). Выход `IdentityResolver`."""

    client_id: str | None = None
    """`None` только у справочных источников (`kind: reference`): у курса валют
    клиента нет. У событийных и профильных источников запись без `client_id`
    в карантине (§34) и сюда не доходит."""

    def debug_row(self) -> dict[str, Any]:
        return {**super().debug_row(), "client_id": self.client_id}


@dataclass(frozen=True)
class TimedRecord(IdentifiedRecord):
    """Запись с нормализованным временем (§12). Выход `TimestampNormalizer`."""

    timestamp_utc: datetime | None = None
    """Абсолютный instant события — для сортировки, cutoff и lineage (§12 п.1)."""

    calendar_timezone: str | None = None
    """IANA-зона для календарных признаков (§12 п.3). Не смещение: смещение
    не помнит исторических правил, а §12.1 требует именно их."""

    processing_time_utc: datetime | None = None
    """Время загрузки. §12.2 разрешает использовать его только для lineage,
    watermark и late-arriving monitoring — но не вместо event time."""

    quality_flags: tuple[QualityFlag, ...] = ()

    def with_flag(self, flag: QualityFlag) -> "TimedRecord":
        """Добавить пометку, сохранив детерминированный порядок флагов."""
        if flag in self.quality_flags:
            return self
        merged = tuple(sorted({*self.quality_flags, flag}))
        return _replace(self, quality_flags=merged)

    def debug_row(self) -> dict[str, Any]:
        return {
            **super().debug_row(),
            "timestamp_utc": _isoformat(self.timestamp_utc),
            "calendar_timezone": self.calendar_timezone,
            "processing_time_utc": _isoformat(self.processing_time_utc),
            "quality_flags": [str(flag) for flag in self.quality_flags],
        }


def _replace(record: TimedRecord, **changes: Any) -> TimedRecord:
    from dataclasses import replace

    return replace(record, **changes)


def _isoformat(moment: datetime | None) -> str | None:
    """ISO-8601 UTC с `Z` — вид §32.2, а не `+00:00` от `isoformat()`."""
    if moment is None:
        return None
    return moment.isoformat().replace("+00:00", "Z")
