"""TimeFeatureBuilder — §25, §25.1, §25.3.

Последний компонент цепочки §37.2 перед выдачей: `timeline → local hour/day`.
Делает он немного — час и день недели по локальной зоне плюс флаг
`lifetime_first`, — но каждое из трёх решений здесь ровно то, на котором
реализации расходятся молча.

**Локальные признаки нельзя считать из UTC** (§25, §12 п.4, QA §35 п.8).
В UTC+5 покупка в 23:30 по Алматы выглядит дневной (18:30 UTC), и признак
«вечерняя активность» превращается в свою противоположность. Поэтому здесь
нет ни одной ветки, где зона берётся не из записи: событие без
`calendar_timezone` — блокирующая ошибка, а не «посчитаем по UTC». Подставить
UTC невозможно, потому что подставлять нечем.

**Зона — имя IANA, а не смещение** (§12.1, QA §35 пп. 9–10). Алматы жил в
UTC+6 до 2024-03-01 и в UTC+5 после; фиксированное `+05:00` дало бы для
истории час, которого не было. `ZoneInfo` знает исторические правила, число —
не знает, поэтому в компонент попадает имя.

**`lifetime_first` не назначается, а выводится** (§5 п.13, §25.1). Флаг
означает «самое раннее известное событие клиента» и берётся из `position == 0`,
проставленной §13. Параметра, которым его можно поставить куда-то ещё, нет.
Условие его правды — полный timeline: если историю обрезать до этого шага,
`position == 0` станет началом окна, а не жизни, и флаг соврёт токенайзеру
(§25.1 отличает `FIRST_EVENT` от `WINDOW_START` именно им). Поэтому первое
встреченное событие клиента обязано иметь `position == 0` — иначе обработка
останавливается.

**`delta_from_previous_event` не считается вовсе.** §25.1 разрешает считать
full-timeline дельту для QA, но её значение относится к событию, которого
после truncation в окне может не быть. Поля, которого нет, нельзя случайно
отдать модели.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .core.debug_dump import DebugDump, Stage
from .feature_projection import PROFILE_SECTION, ProjectedRecord
from .schema.event import CalendarTimeFeatures
from .timeline_builder import TimelineRecord

COMPONENT = "time_feature_builder"


class TimeFeatureError(RuntimeError):
    """Событие, для которого календарный признак посчитать нечем."""


@dataclass(frozen=True)
class TimeFeaturedRecord(TimelineRecord):
    """Событие с локальными календарными признаками (§5 п.9, §25)."""

    calendar_time_features: CalendarTimeFeatures | None = None
    """Час и день недели в бизнес-зоне. Отдельный атрибут, а не запись в
    `fields`: §25.3 требует, чтобы time metadata не токенизировалась как
    обычные key/value и не расходовала `max_tokens_per_event`."""

    lifetime_first: bool = False
    """§5 п.13: `true` только на самом раннем известном событии клиента."""

    def debug_row(self) -> dict[str, Any]:
        features = self.calendar_time_features
        return {
            **super().debug_row(),
            "hour_of_day_local": features.hour_of_day_local if features else None,
            "day_of_week_local": features.day_of_week_local if features else None,
            "lifetime_first": self.lifetime_first,
        }


@dataclass
class TimeFeatureReport:
    """Что получилось из календарных признаков."""

    events: int = 0
    clients: int = 0
    lifetime_first: int = 0
    hours_shifted_from_utc: int = 0
    """Событий, у которых локальный час не совпал с часом UTC.

    Не статистика, а детектор. Ровно этот счётчик уходит в ноль, если кто-то
    начнёт считать признаки напрямую из UTC — то есть сделает то, что
    запрещают §25 и QA §35 п.8. Ноль на данных, где зона не UTC, означает,
    что зона не применяется."""

    days_shifted_from_utc: int = 0
    """Событий, у которых локальный день недели не совпал с днём UTC.
    Ночные события у смещённой зоны уезжают на сутки — их и считаем."""

    def merge(self, other: "TimeFeatureReport") -> None:
        self.events += other.events
        self.clients += other.clients
        self.lifetime_first += other.lifetime_first
        self.hours_shifted_from_utc += other.hours_shifted_from_utc
        self.days_shifted_from_utc += other.days_shifted_from_utc

    def summary(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "clients": self.clients,
            "lifetime_first": self.lifetime_first,
            "hours_shifted_from_utc": self.hours_shifted_from_utc,
            "days_shifted_from_utc": self.days_shifted_from_utc,
        }


class TimeFeatureBuilder:
    """Локальные календарные признаки и `lifetime_first` (§25)."""

    def __init__(self, *, monitor=None, debug: DebugDump | None = None) -> None:
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self._zones: dict[str, ZoneInfo] = {}
        self.report = TimeFeatureReport()

    def build(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        """Достроить события календарными признаками (§25).

        Профили проходят насквозь: профиль привязан к T целиком, локального
        часа у него нет.
        """
        tracing = self._debug.enabled
        started: set[str] = set()

        for record in records:
            if record.schema_section == PROFILE_SECTION or record.event_type is None:
                yield record
                continue

            self._check_computable(record, started)
            zone = self._zone(record)
            local = record.timestamp_utc.astimezone(zone)
            features = CalendarTimeFeatures(
                hour_of_day_local=local.hour,
                # §5 п.12: понедельник = 0. `weekday()` даёт именно эту
                # конвенцию, `isoweekday()` — сдвинутую на единицу.
                day_of_week_local=local.weekday(),
            )
            result = _with_time_features(record, features, record.position == 0)

            self.report.events += 1
            if result.lifetime_first:
                self.report.lifetime_first += 1
            if local.hour != record.timestamp_utc.hour:
                self.report.hours_shifted_from_utc += 1
            if local.weekday() != record.timestamp_utc.weekday():
                self.report.days_shifted_from_utc += 1

            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _check_computable(self, record: ProjectedRecord, started: set[str]) -> None:
        """Чего не хватает, чтобы признак был посчитан честно."""
        if record.timestamp_utc is None:
            raise TimeFeatureError(
                f"{record.raw_reference}: событие без timestamp_utc — §12 обязан был "
                "отправить его в карантин"
            )
        if not record.calendar_timezone:
            # §25 и QA §35 п.8: подстановки UTC здесь нет и быть не может —
            # она превратила бы вечернее событие в дневное и не оставила следа.
            raise TimeFeatureError(
                f"{record.raw_reference}: событие без calendar_timezone — считать "
                "локальный час не из чего, а из UTC нельзя (§25, §12 п.4)"
            )
        if not record.event_id:
            # §25.3: календарные признаки связываются с событием через event_id.
            raise TimeFeatureError(
                f"{record.raw_reference}: событие без event_id — §25.3 связывает "
                "календарные признаки именно с ним"
            )
        if getattr(record, "position", None) is None:
            raise TimeFeatureError(
                f"{record.raw_reference}: событие без position — §25.1 выводит "
                "lifetime_first из порядка §13, а его здесь ещё не было"
            )

        client_id = record.client_id or ""
        if client_id not in started:
            # Счётчик поднимается здесь, а не в конце: `build` — генератор, и
            # итог, посчитанный после цикла, у недочитанного потока остался бы
            # нулём, ничем себя не выдав.
            started.add(client_id)
            self.report.clients += 1
            if record.position != 0:
                # Обрезка истории до этого шага сделала бы lifetime_first ложью:
                # начало окна выдавалось бы за начало жизни клиента (§25.1).
                raise TimeFeatureError(
                    f"{record.raw_reference}: первое событие клиента с position="
                    f"{record.position}, а не 0 — timeline обрезан до §25, и "
                    "lifetime_first отличал бы WINDOW_START от FIRST_EVENT неверно"
                )

    def _zone(self, record: ProjectedRecord) -> ZoneInfo:
        """Разобрать IANA-зону записи, запомнив разбор.

        Зон в потоке единицы, а событий десятки тысяч; разбор кэшируется,
        чтобы стоимость исторических правил платилась один раз на зону.
        """
        name = record.calendar_timezone
        zone = self._zones.get(name)
        if zone is None:
            try:
                zone = ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise TimeFeatureError(
                    f"{record.raw_reference}: {name!r} не является IANA timezone — "
                    "§12 обязан был отсечь такую запись"
                ) from error
            self._zones[name] = zone
        return zone


def _with_time_features(
    record: ProjectedRecord, features: CalendarTimeFeatures, lifetime_first: bool
) -> TimeFeaturedRecord:
    """Достроить запись, не потеряв уже накопленного.

    Поля родителя переносятся через `dataclasses.fields`, а не перечислением:
    перечисление молча потеряло бы атрибут, добавленный в `TimelineRecord`
    позже, и запись доехала бы до выдачи обеднённой — без ошибки и без следа.
    """
    inherited = {f.name: getattr(record, f.name) for f in dataclass_fields(record)}
    return TimeFeaturedRecord(
        **inherited,
        calendar_time_features=features,
        lifetime_first=lifetime_first,
    )
