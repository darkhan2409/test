"""TimelineBuilder — §13, §26.

Самая важная сортировка в проекте: порядок событий и есть то, что читает
модель. Если он поедет, поедет всё — но ни один счётчик этого не заметит,
потому что записи все на месте.

Ключ сортировки состоит из трёх частей (§13), и каждая нужна ровно потому,
что предыдущей не хватает:

1. `timestamp_utc` — основной порядок;
2. `source_priority` — когда два события произошли в один момент в разных
   системах. Ранг объявлен в Source Contract (§4) и не зависит ни от воркера,
   ни от файла;
3. `source_record_id` — когда и источник один. Здесь уже нет ничего, кроме
   самого идентификатора записи, и это последний рубеж: без него порядок
   определяла бы очередь поступления, то есть нарезка потока по воркерам.

`ordering_key` собирается строкой и кладётся в событие (§5 п.7): он делает
принятое решение видимым в выходных данных, а не спрятанным в коде
сортировщика.

Timeline сохраняется **целиком** (§26 п.7). Обрезка окна — работа токенайзера,
и делать её здесь значило бы решать за него, сколько истории он увидит.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from .core.debug_dump import DebugDump, Stage
from .feature_projection import PROFILE_SECTION, ProjectedRecord
from .schema.source_contract import SourceContractRegistry

COMPONENT = "timeline_builder"

# Ширина числового поля в `ordering_key`. Ранг источника печатается с
# ведущими нулями, иначе строковое сравнение поставило бы `10` перед `9`.
_PRIORITY_WIDTH = 6


class TimelineError(RuntimeError):
    """Событие, которое нельзя разместить в timeline, — блокирующая ошибка."""


@dataclass(frozen=True)
class TimelineRecord(ProjectedRecord):
    """Событие с местом в timeline клиента (§5 п.7, §13)."""

    ordering_key: str | None = None
    position: int | None = None
    """Номер события в timeline клиента, с нуля. Нужен для отладки и для
    §25.1 (`lifetime_first` — событие с позицией 0)."""

    def debug_row(self) -> dict[str, Any]:
        return {**super().debug_row(), "ordering_key": self.ordering_key,
                "position": self.position}


@dataclass
class TimelineReport:
    """Что получилось из событий."""

    clients: int = 0
    events: int = 0
    ties_by_priority: int = 0
    ties_by_record_id: int = 0

    def merge(self, other: "TimelineReport") -> None:
        self.clients += other.clients
        self.events += other.events
        self.ties_by_priority += other.ties_by_priority
        self.ties_by_record_id += other.ties_by_record_id

    def summary(self) -> dict[str, Any]:
        return {
            "clients": self.clients,
            "events": self.events,
            "ties_resolved_by_source_priority": self.ties_by_priority,
            "ties_resolved_by_source_record_id": self.ties_by_record_id,
        }


class TimelineBuilder:
    """Единый timeline клиента (§26) с детерминированным порядком (§13)."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        *,
        cutoff: datetime,
        monitor=None,
        debug: DebugDump | None = None,
    ) -> None:
        self.registry = registry
        self.cutoff = cutoff
        self._priority = registry.source_priority()
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = TimelineReport()

    def build(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        """Разложить события по клиентам и выдать их в порядке §13.

        Профили и записи без клиента проходят насквозь: timeline — про
        события, а профиль привязан к T целиком.
        """
        tracing = self._debug.enabled

        events: dict[str, list[ProjectedRecord]] = {}
        passthrough: list[ProjectedRecord] = []

        for record in records:
            if record.schema_section == PROFILE_SECTION or record.event_type is None:
                passthrough.append(record)
                continue
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])
            self._check_placeable(record)
            events.setdefault(record.client_id or "", []).append(record)

        yield from passthrough

        for client_id in sorted(events):
            ordered = self._order(events[client_id])
            self.report.clients += 1
            self.report.events += len(ordered)
            for record in ordered:
                if tracing:
                    self._debug.record(COMPONENT, Stage.OUT, [record.debug_row()])
                yield record

    def _check_placeable(self, record: ProjectedRecord) -> None:
        """Событие без времени, клиента или идентификатора разместить нельзя."""
        if record.timestamp_utc is None:
            raise TimelineError(
                f"{record.raw_reference}: событие без timestamp_utc — §12 обязан был "
                "отправить его в карантин"
            )
        if record.timestamp_utc > self.cutoff:
            # §26 п.5: все события timeline не позже T.
            raise TimelineError(
                f"{record.raw_reference}: {record.timestamp_utc.isoformat()} позже T — "
                "утечка мимо §14"
            )
        if not record.client_id:
            raise TimelineError(f"{record.raw_reference}: событие без client_id (§7)")
        if not record.source_record_id:
            raise TimelineError(
                f"{record.raw_reference}: событие без source_record_id — третий ключ "
                "tie-break §13 нечем заполнить"
            )

    def _order(self, records: list[ProjectedRecord]) -> list[TimelineRecord]:
        ordered = sorted(records, key=self._sort_key)
        self._count_ties(ordered)
        return [
            _with_key(record, self.ordering_key(record), position)
            for position, record in enumerate(ordered)
        ]

    def _sort_key(self, record: ProjectedRecord) -> tuple[datetime, int, str]:
        """Ключ §13 — три части, каждая на случай, когда предыдущей мало."""
        return (
            record.timestamp_utc,
            self._priority[record.source],
            record.source_record_id,
        )

    def _count_ties(self, ordered: list[TimelineRecord | ProjectedRecord]) -> None:
        """Посчитать, чем разрешались совпадения времени.

        Не ради статистики: нулевые счётчики на данных, где tie-break заведомо
        есть, означают, что проверка порядка ничего не проверяет.
        """
        for previous, current in zip(ordered, ordered[1:]):
            if previous.timestamp_utc != current.timestamp_utc:
                continue
            if self._priority[previous.source] != self._priority[current.source]:
                self.report.ties_by_priority += 1
            else:
                self.report.ties_by_record_id += 1

    def ordering_key(self, record: ProjectedRecord) -> str:
        """Детерминированный ключ сортировки строкой (§5 п.7, §13).

        Кладётся в событие, чтобы принятое решение было видно в выходных
        данных, а не выводилось заново читателем. Составные части печатаются
        фиксированной ширины: при строковом сравнении `10` иначе оказалось бы
        раньше `9`.
        """
        moment = record.timestamp_utc.isoformat().replace("+00:00", "Z")
        priority = f"{self._priority[record.source]:0{_PRIORITY_WIDTH}d}"
        return f"{moment}|{priority}|{record.source_record_id}"


def _with_key(record: ProjectedRecord, ordering_key: str, position: int) -> TimelineRecord:
    return TimelineRecord(
        source=record.source,
        partition=record.partition,
        line_number=record.line_number,
        source_record_id=record.source_record_id,
        source_schema_version=record.source_schema_version,
        client_ref=record.client_ref,
        payload=record.payload,
        client_id=record.client_id,
        timestamp_utc=record.timestamp_utc,
        calendar_timezone=record.calendar_timezone,
        processing_time_utc=record.processing_time_utc,
        quality_flags=record.quality_flags,
        event_type=record.event_type,
        event_id=record.event_id,
        fields=record.fields,
        schema_section=record.schema_section,
        ordering_key=ordering_key,
        position=position,
    )
