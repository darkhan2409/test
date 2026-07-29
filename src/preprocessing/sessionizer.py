"""Sessionizer — §20, §20.1, §20.2, §20.3.

Строка app-лога сама по себе не событие: «открыл экран» ничего не говорит о
клиенте. Событие — сессия целиком, и §20 склеивает строки в `APP_SESSION`.

Границы сессии (§20.1): смена клиента, разрыв больше `session_gap`, явная
граница. Явной границей объявлен `LOGIN`: вход в приложение — новая сессия
по смыслу, даже если предыдущая формально не истекла.

Про будущее (§20.3, §14.1). Действия после T сюда не доходят: отсечка (§14)
стоит раньше по цепочке §37.2, и до сессионизации доживает только доступная
часть. Поэтому длительность и число экранов считаются по ней автоматически —
не потому, что здесь стоит отдельная проверка, а потому, что будущего в
данных уже нет. Сессия, пересекавшая T, приходит сюда обрезанной и выглядит
как обычная короткая.

Компонент собирающий: действия одного клиента разбросаны по партициям
месяцев, и границу сессии видно только на полном наборе. Порядок выдачи —
каноническое место первой строки сессии, поэтому поток остаётся стабильным.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.debug_dump import DebugDump, Stage
from .core.hashing import event_id as make_event_id
from .core.monitor import DataQualityMonitor, Metric, Total
from .event_mapper import EventMapping, MappedRecord
from .records import TimedRecord
from .schema.source_contract import SourceContractRegistry

COMPONENT = "sessionizer"
SESSION_EVENT_TYPE = "APP_SESSION"


class SessionizationError(RuntimeError):
    """Ошибка конфигурации сессионизации — блокирующая."""


class SessionizationConfig(BaseModel):
    """Версионируемый конфиг сессионизации (§20.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sessionization_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    action_field: str = Field(min_length=1)
    screen_field: str = Field(min_length=1)
    session_start_actions: tuple[str, ...] = Field(min_length=1)
    session_actions: tuple[str, ...] = Field(min_length=1)
    device_category_field: str = Field(min_length=1)
    app_version_field: str = Field(min_length=1)
    max_session_duration_hours: int = Field(gt=0)

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_sessionization_config(
    path: Path, registry: SourceContractRegistry, event_mapping: EventMapping
) -> SessionizationConfig:
    """Загрузить конфиг и сверить его с контрактом и маппингом событий."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SessionizationError(f"{path}: ожидался YAML-объект")
    config = SessionizationConfig.model_validate(document)

    contract = registry.contract(config.source)
    for label, column in (
        ("action_field", config.action_field),
        ("screen_field", config.screen_field),
        ("device_category_field", config.device_category_field),
        ("app_version_field", config.app_version_field),
    ):
        if column not in contract.columns:
            raise SessionizationError(
                f"{config.source}: {label}={column!r} нет в схеме источника"
            )

    if not set(config.session_start_actions) <= set(config.session_actions):
        raise SessionizationError(
            "session_start_actions обязаны быть подмножеством session_actions"
        )

    _check_agrees_with_event_mapping(config, event_mapping)

    if SESSION_EVENT_TYPE not in event_mapping.event_types:
        raise SessionizationError(
            f"{SESSION_EVENT_TYPE} не входит в утверждённые event_types (§10 п.2)"
        )

    return config


def _check_agrees_with_event_mapping(
    config: SessionizationConfig, event_mapping: EventMapping
) -> None:
    """Сверить два конфига, описывающих одно множество с разных сторон.

    Маппинг событий помечает действия app-лога как «не событие» (§10 п.5),
    сессионизация собирает их в `APP_SESSION`. Расхождение ломает данные
    тремя разными способами, и сообщение обязано называть настоящий —
    иначе чинить будут не то.
    """
    codes = event_mapping.sources[config.source].codes
    declared = set(config.session_actions)
    technical = {code for code, event_type in codes.items() if event_type is None}

    problems: list[str] = []

    lost = sorted(technical - declared)
    if lost:
        problems.append(
            f"{lost} — маппинг оставил их без типа события, а сессионизация про них "
            "не знает: строки исчезнут молча, не став ни событием, ни частью сессии"
        )

    promoted = sorted(code for code in declared & set(codes) if codes[code] is not None)
    if promoted:
        problems.append(
            f"{promoted} — маппинг делает их событием, а сессионизация собирает "
            "в APP_SESSION: строка попадёт в timeline дважды"
        )

    unmapped = sorted(declared - set(codes))
    if unmapped:
        problems.append(
            f"{unmapped} — объявлены в сессии, но в маппинге их нет вовсе: "
            "EventMapper отправит их в карантин как неизвестный код (§10 п.3)"
        )

    if problems:
        raise SessionizationError(
            "расхождение sessionization.yaml и event_mapping.yaml: " + "; ".join(problems)
        )


@dataclass
class SessionReport:
    """Что получилось из app-логов."""

    actions_consumed: int = 0
    sessions_built: int = 0
    empty_sessions: int = 0
    over_long_sessions: int = 0
    over_limit_screens: int = 0

    def merge(self, other: "SessionReport") -> None:
        self.actions_consumed += other.actions_consumed
        self.sessions_built += other.sessions_built
        self.empty_sessions += other.empty_sessions
        self.over_long_sessions += other.over_long_sessions
        self.over_limit_screens += other.over_limit_screens

    def summary(self) -> dict[str, Any]:
        return {
            "actions_consumed": self.actions_consumed,
            "sessions_built": self.sessions_built,
            "empty_sessions": self.empty_sessions,
            "over_long_sessions": self.over_long_sessions,
            "sessions_over_screen_limit": self.over_limit_screens,
        }


class Sessionizer:
    """Склейка app-логов в события `APP_SESSION`."""

    def __init__(
        self,
        config: SessionizationConfig,
        *,
        session_gap: timedelta,
        max_values_per_field: int,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.config = config
        self.session_gap = session_gap
        self.max_values_per_field = max_values_per_field
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = SessionReport()

    def sessionize(self, records: Iterable[MappedRecord]) -> Iterator[MappedRecord]:
        tracing = self._debug.enabled

        passthrough: list[MappedRecord] = []
        actions: defaultdict[str, list[MappedRecord]] = defaultdict(list)

        for record in records:
            if self._is_session_action(record):
                if tracing:
                    self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])
                actions[record.client_id or ""].append(record)
            else:
                passthrough.append(record)

        built = [
            session
            for client in sorted(actions)
            for session in self._sessions_of(actions[client])
        ]
        self.report.actions_consumed = sum(len(group) for group in actions.values())
        self.report.sessions_built = len(built)

        for record in sorted(passthrough + built, key=_canonical_position):
            if tracing and record.event_type == SESSION_EVENT_TYPE:
                self._debug.record(COMPONENT, Stage.OUT, [record.debug_row()])
            yield record

    def _is_session_action(self, record: MappedRecord) -> bool:
        """Строка app-лога, которую §10 п.5 оставил без типа события.

        Push-уведомление сюда не попадает: у него есть свой `event_type`,
        оно получено клиентом, а не совершено им, и в сессию не входит.
        """
        return record.source == self.config.source and record.event_type is None

    # ------------------------------------------------------------------ #
    # Границы сессий
    # ------------------------------------------------------------------ #

    def _sessions_of(self, actions: list[MappedRecord]) -> list[MappedRecord]:
        """Разрезать действия одного клиента на сессии (§20.1)."""
        ordered = sorted(actions, key=_action_order)
        sessions: list[MappedRecord] = []
        current: list[MappedRecord] = []

        for action in ordered:
            if current and self._starts_new_session(current[-1], action):
                sessions.append(self._build_session(current))
                current = []
            current.append(action)

        if current:
            sessions.append(self._build_session(current))
        return sessions

    def _starts_new_session(self, previous: MappedRecord, action: MappedRecord) -> bool:
        if str(action.payload.get(self.config.action_field)) in self.config.session_start_actions:
            return True
        if previous.timestamp_utc is None or action.timestamp_utc is None:
            return True
        return action.timestamp_utc - previous.timestamp_utc > self.session_gap

    # ------------------------------------------------------------------ #
    # Сборка события
    # ------------------------------------------------------------------ #

    def _build_session(self, actions: list[MappedRecord]) -> MappedRecord:
        first, last = actions[0], actions[-1]
        if first.timestamp_utc is None or last.timestamp_utc is None:
            raise SessionizationError(
                f"{first.raw_reference}: действие без времени дошло до сессионизации — "
                "§12 обязан был отправить его в карантин"
            )
        self._monitor.add_total(Total.SESSIONS_BUILT)

        screens = self._screens(actions)
        duration = last.timestamp_utc - first.timestamp_utc
        payload = {
            "session_start": first.timestamp_utc.isoformat().replace("+00:00", "Z"),
            "duration_seconds": int(duration.total_seconds()),
            "action_count": len(actions),
            "screens_count": len(screens.unique),
            "screens": list(screens.unique),
            "first_screen": screens.first,
            "last_screen": screens.last,
            "device_category": first.payload.get(self.config.device_category_field),
            "app_version": first.payload.get(self.config.app_version_field),
        }
        # `top_intent` из §20.2 не считается: он требует утверждённого
        # справочника «экран → намерение», которого у этих логов нет.
        # Догадка вместо бизнес-классификации хуже отсутствующего поля.

        self._count_anomalies(payload, screens)

        return MappedRecord(
            source=first.source,
            partition=first.partition,
            line_number=first.line_number,
            # Идентификатор сессии — идентификатор её первой строки: он
            # реальный, стабильный и сохраняет lineage до конкретной записи
            # источника (§8). Синтетический ключ такой связи не дал бы.
            source_record_id=first.source_record_id,
            source_schema_version=first.source_schema_version,
            client_ref=first.client_ref,
            payload=payload,
            client_id=first.client_id,
            timestamp_utc=first.timestamp_utc,
            calendar_timezone=first.calendar_timezone,
            processing_time_utc=None,
            quality_flags=first.quality_flags,
            event_type=SESSION_EVENT_TYPE,
            # Идентификатор считается здесь, а не в EventMapper: событие
            # родилось на этом шаге. Формула та же (§8) и тот же `hash_policy`
            # (§29.1) — иначе у части событий идентификаторы жили бы по своим
            # правилам.
            event_id=make_event_id(
                source_system=first.source,
                source_record_id=first.source_record_id,
                event_type=SESSION_EVENT_TYPE,
                event_timestamp=first.timestamp_utc,
            ),
        )

    def _screens(self, actions: list[MappedRecord]) -> "_Screens":
        """Экраны сессии в хронологическом порядке первого появления.

        Порядок именно хронологический, а не алфавитный: §20.3 требует
        детерминированности, а §21 для экранов — значимости порядка. Сортировка
        по алфавиту сделала бы порядок детерминированным, но неверным.

        Список отдаётся **целиком**. Обрезкой по `max_values_per_field`
        владеет §21 (компонент 2.14) — одно правило, одно место. Здесь лимит
        известен только затем, чтобы поднять аномалию «too many values»
        (§33.13): знать лимит и применять его — разные вещи.
        """
        seen: list[str] = []
        for action in actions:
            screen = action.payload.get(self.config.screen_field)
            if isinstance(screen, str) and screen and screen not in seen:
                seen.append(screen)

        if len(seen) > self.max_values_per_field:
            self.report.over_limit_screens += 1

        return _Screens(
            unique=tuple(seen),
            first=seen[0] if seen else None,
            last=seen[-1] if seen else None,
        )

    def _count_anomalies(self, payload: dict[str, Any], screens: "_Screens") -> None:
        """§33.13: negative/excessive duration, empty session, too many values."""
        anomaly = False

        if not screens.unique:
            # Сессия без единого экрана: клиент вошёл и ничего не открыл.
            self.report.empty_sessions += 1
            anomaly = True

        duration = payload["duration_seconds"]
        if duration < 0:
            anomaly = True
        if duration > self.config.max_session_duration_hours * 3600:
            self.report.over_long_sessions += 1
            anomaly = True

        if len(screens.unique) > self.max_values_per_field:
            anomaly = True

        if anomaly:
            self._monitor.count(Metric.SESSION_ANOMALY_RATE)


@dataclass(frozen=True)
class _Screens:
    unique: tuple[str, ...]
    first: str | None
    last: str | None


def _canonical_position(record: TimedRecord) -> tuple[str, int]:
    return (record.partition, record.line_number)


def _action_order(record: MappedRecord) -> tuple[Any, ...]:
    """Порядок действий внутри клиента.

    Время — первый ключ, `source_record_id` — второй: два действия в одну
    секунду иначе разошлись бы по порядку файлов, а он от числа воркеров
    зависеть не должен (§29 п.8).
    """
    return (record.timestamp_utc, record.source_record_id)
