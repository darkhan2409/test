"""TimestampNormalizer — §12, §12.1, §12.2.

Из трёх разных форматов времени получается одно и то же: абсолютный instant
`timestamp_utc` и IANA-зона `calendar_timezone`. Это два разных представления
одного момента, и §12 требует хранить оба: по первому сортируют и отсекают,
по второму считают «час дня» и «день недели».

Почему нельзя обойтись одним. Мгновение перевода Казахстана на UTC+05
(1 марта 2024) означает, что один и тот же UTC-час до и после даёт разный
локальный час. Считать `hour_of_day_local` из UTC — значит сдвинуть всё
поведение клиента на час в одной половине истории (§12 п.4, §12.1).

Зона берётся тремя способами, и каждый объявлен в Source Contract:

- `from_field_mapping` — из поля записи через справочник (core_payments);
- `client_profile_region` — из региона клиента в его профиле на T;
- `source_default` / `utc` — утверждённое умолчание источника.

Второй способ — единственное место, где компонент смотрит за пределы своей
записи. Индекс зон клиентов строится один раз до основного прохода из
`profile_snapshots`; иначе календарные признаки карточных и app-событий
пришлось бы считать по умолчанию для всей страны, что §12.1 прямо запрещает.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .core.quarantine import Quarantine, ReasonCode
from .records import IdentifiedRecord, QualityFlag, TimedRecord
from .schema.source_contract import (
    SourceContract,
    SourceContractRegistry,
    SourceKind,
    TimestampKind,
    TimezonePolicy,
)

COMPONENT = "timestamp_normalizer"
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Момент, откуда берётся локальная дата для источников с датой вместо времени
# (снимок профиля, курс валют). Начало суток в зоне источника: снимок за
# 1 января относится к началу этого дня, а не к произвольному его часу.
START_OF_DAY = (0, 0, 0)


class TimestampPolicyError(RuntimeError):
    """Ошибка политики времени — блокирующая, до обработки записей."""


class ClientRegionSource(BaseModel):
    """Откуда берётся регион клиента для политики `client_profile_region`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    field: str = Field(min_length=1)
    mapping: str = Field(min_length=1)


class TimestampPolicy(BaseModel):
    """Политика §12: разбор времени и выбор календарной зоны."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_policy_version: str = Field(min_length=1)
    calendar_timezone_policy_version: str = Field(min_length=1)
    future_horizon_days: int = Field(ge=0)
    timezone_mappings: dict[str, dict[str, str]] = Field(min_length=1)
    client_profile_region: ClientRegionSource

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_timestamp_policy(path: Path, registry: SourceContractRegistry) -> TimestampPolicy:
    """Загрузить политику и проверить её против Source Contracts."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TimestampPolicyError(f"{path}: ожидался YAML-объект")
    policy = TimestampPolicy.model_validate(document)

    for name in sorted(policy.timezone_mappings):
        for value, zone in sorted(policy.timezone_mappings[name].items()):
            _zone_or_raise(zone, f"справочник {name}, значение {value}")

    # Контракт ссылается на справочник по имени — имя без справочника означает
    #, что зона не определится ни у одной записи источника.
    for name in sorted(registry.sources):
        contract = registry.sources[name]
        required = contract.timezone.mapping
        if required is not None and required not in policy.timezone_mappings:
            raise TimestampPolicyError(
                f"{name}: контракт ссылается на справочник зон {required!r}, "
                "которого нет в политике времени"
            )
        if contract.timezone.default is not None:
            _zone_or_raise(contract.timezone.default, f"source default источника {name}")

    _check_client_region_source(policy, registry)
    return policy


def _check_client_region_source(
    policy: TimestampPolicy, registry: SourceContractRegistry
) -> None:
    """Проверить объявление источника региона клиента.

    Если оно указывает в пустоту, все источники с политикой
    `client_profile_region` молча уедут на source default — то есть на единый
    UTC+05 для всей страны, что §12.1 запрещает.
    """
    declared = policy.client_profile_region
    needed = any(
        contract.timezone.policy is TimezonePolicy.CLIENT_PROFILE_REGION
        for contract in registry.sources.values()
    )
    if not needed:
        return

    contract = registry.sources.get(declared.source)
    if contract is None:
        raise TimestampPolicyError(
            f"client_profile_region: источника {declared.source!r} нет в Source Contracts"
        )
    if contract.kind is not SourceKind.PROFILE:
        raise TimestampPolicyError(
            f"client_profile_region: {declared.source} имеет kind {contract.kind}, "
            "регион клиента берётся из профильного источника"
        )
    if declared.field not in contract.columns:
        raise TimestampPolicyError(
            f"client_profile_region: у {declared.source} нет колонки {declared.field!r}"
        )
    if declared.mapping not in policy.timezone_mappings:
        raise TimestampPolicyError(
            f"client_profile_region: справочника {declared.mapping!r} нет в политике"
        )
    # Иначе индекс зон клиентов зависел бы сам от себя.
    _fixed_source_zone(contract)


def _zone_or_raise(zone: str, where: str) -> ZoneInfo:
    try:
        return ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise TimestampPolicyError(
            f"{where}: {zone!r} не является IANA-зоной ({error}). "
            "На Windows проверьте пакет tzdata."
        ) from None


# --------------------------------------------------------------------------- #
# Индекс зон клиентов
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClientTimezoneIndex:
    """Календарная зона клиента на момент T.

    Строится из `profile_snapshots`: берётся последний снимок с датой не
    позже T (§6 п.1) и его регион. Клиенты без снимка сюда не попадают —
    у них применяется source default и поднимается
    `calendar_timezone_fallback_rate` (§33.11).
    """

    zones: Mapping[str, str]

    @classmethod
    def build(
        cls,
        records: Iterable[IdentifiedRecord],
        *,
        registry: SourceContractRegistry,
        policy: TimestampPolicy,
        cutoff: datetime,
    ) -> "ClientTimezoneIndex":
        declared = policy.client_profile_region
        table = policy.timezone_mappings[declared.mapping]
        contract = registry.contract(declared.source)
        # Ровно та же зона, в которой основной проход превращает дату снимка
        # в instant. Иначе один и тот же снимок оказался бы по разные стороны
        # от T здесь и там: разница между полуночью UTC и полуночью Алматы —
        # пять часов, и в них попадают снимки первого числа месяца.
        source_zone = _fixed_source_zone(contract)
        latest: dict[str, tuple[date, str]] = {}

        for record in records:
            if record.source != declared.source or record.client_id is None:
                continue

            snapshot = _snapshot_date(record, contract)
            if snapshot is None:
                continue
            # Снимок позже T использовать нельзя (§6 п.1, §14): зона клиента
            # на T не может опираться на то, что стало известно после T.
            if _localize(datetime.combine(snapshot, datetime.min.time()), source_zone) > cutoff:
                continue

            region = record.payload.get(declared.field)
            zone = table.get(region) if isinstance(region, str) else None
            if zone is None:
                continue

            known = latest.get(record.client_id)
            if known is None or snapshot > known[0]:
                latest[record.client_id] = (snapshot, zone)

        return cls(zones={client: zone for client, (_, zone) in sorted(latest.items())})

    def zone_of(self, client_id: str | None) -> str | None:
        if client_id is None:
            return None
        return self.zones.get(client_id)


def _fixed_source_zone(contract: SourceContract) -> str:
    """Зона источника, не зависящая от содержимого записи.

    Только такая и годится для построения индекса: зона, выводимая из данных
    клиента, потребовала бы индекса, который ещё не построен.
    """
    if contract.timezone.policy is TimezonePolicy.UTC:
        return "UTC"
    if contract.timezone.policy is TimezonePolicy.SOURCE_DEFAULT:
        return contract.timezone.default or "UTC"
    raise TimestampPolicyError(
        f"{contract.name}: политика зоны {contract.timezone.policy} зависит от записи, "
        "по такому источнику нельзя строить индекс зон клиентов"
    )


def _snapshot_date(record: IdentifiedRecord, contract: SourceContract) -> date | None:
    value = record.payload.get(contract.event_time.field)
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Нормализация
# --------------------------------------------------------------------------- #


class TimestampNormalizer:
    """Приведение времени к UTC и выбор календарной зоны."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        policy: TimestampPolicy,
        *,
        cutoff: datetime,
        monitor: DataQualityMonitor,
        quarantine: Quarantine,
        client_zones: ClientTimezoneIndex | None = None,
        debug: DebugDump | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.cutoff = cutoff
        self.client_zones = client_zones or ClientTimezoneIndex(zones={})
        self._monitor = monitor
        self._quarantine = quarantine
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self._future_after = cutoff + timedelta(days=policy.future_horizon_days)

    def normalize(self, records: Iterable[IdentifiedRecord]) -> Iterator[TimedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            timed = self._normalize_one(record)
            if timed is None:
                continue
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [timed.debug_row()])
            yield timed

    def _normalize_one(self, record: IdentifiedRecord) -> TimedRecord | None:
        contract = self.registry.contract(record.source)

        self._monitor.add_total(Total.TIMEZONES_RESOLVED)
        zone_name, fallback = self._resolve_zone(record, contract)
        if zone_name is None:
            self._quarantine.add(
                ReasonCode.UNKNOWN_TIMEZONE,
                source=record.source,
                raw_reference=record.raw_reference,
                partition=record.partition,
                detail=self._zone_detail(contract, record),
            )
            return None

        self._monitor.add_total(Total.TIMESTAMPS_PARSED)
        moment = self._parse(record.payload.get(contract.event_time.field), contract.event_time, zone_name)
        if moment is None:
            # §12 п.8: подставить processing time вместо отсутствующего
            # event time нельзя — событие без своего времени неразмещаемо
            # в timeline, и это карантин, а не починка.
            self._quarantine.add(
                ReasonCode.MISSING_EVENT_TIME,
                source=record.source,
                raw_reference=record.raw_reference,
                partition=record.partition,
                detail=(
                    f"{contract.event_time.field}="
                    f"{record.payload.get(contract.event_time.field)!r} не разбирается "
                    f"как {contract.event_time.kind}"
                ),
            )
            return None

        processing = None
        if contract.processing_time is not None:
            processing = self._parse(
                record.payload.get(contract.processing_time.field),
                contract.processing_time,
                zone_name,
            )

        timed = TimedRecord(
            source=record.source,
            partition=record.partition,
            line_number=record.line_number,
            source_record_id=record.source_record_id,
            source_schema_version=record.source_schema_version,
            client_ref=record.client_ref,
            payload=record.payload,
            client_id=record.client_id,
            timestamp_utc=moment,
            calendar_timezone=zone_name,
            processing_time_utc=processing,
        )

        if fallback:
            self._monitor.count(Metric.CALENDAR_TIMEZONE_FALLBACK_RATE, label=record.source)
            timed = timed.with_flag(QualityFlag.TIMEZONE_FALLBACK)
        self._monitor.add_label_total(Metric.CALENDAR_TIMEZONE_FALLBACK_RATE, record.source)

        if moment > self._future_after:
            self._monitor.count(Metric.TIMESTAMP_ERROR_RATE)
            timed = timed.with_flag(QualityFlag.FUTURE_TIMESTAMP)

        if self._is_late(timed, contract):
            self._monitor.count(Metric.LATE_ARRIVING_RATE, label=record.source)
            timed = timed.with_flag(QualityFlag.LATE_ARRIVING)
        self._monitor.add_label_total(Metric.LATE_ARRIVING_RATE, record.source)

        return timed

    # ------------------------------------------------------------------ #
    # Зона
    # ------------------------------------------------------------------ #

    def _resolve_zone(
        self, record: IdentifiedRecord, contract: SourceContract
    ) -> tuple[str | None, bool]:
        """Имя зоны и признак того, что она взята из умолчания."""
        policy = contract.timezone.policy

        if policy is TimezonePolicy.UTC:
            return "UTC", False

        if policy is TimezonePolicy.SOURCE_DEFAULT:
            return contract.timezone.default, False

        if policy is TimezonePolicy.FROM_FIELD_MAPPING:
            table = self.policy.timezone_mappings[contract.timezone.mapping or ""]
            value = record.payload.get(contract.timezone.field or "")
            zone = table.get(value) if isinstance(value, str) else None
            if zone is not None:
                return zone, False
            # §12 п.7: неизвестная зона — карантин либо утверждённое
            # умолчание источника. У этого источника умолчания нет.
            return contract.timezone.default, contract.timezone.default is not None

        # client_profile_region: зона клиента с его профиля на T.
        zone = self.client_zones.zone_of(record.client_id)
        if zone is not None:
            return zone, False
        return contract.timezone.default, contract.timezone.default is not None

    @staticmethod
    def _zone_detail(contract: SourceContract, record: IdentifiedRecord) -> str:
        if contract.timezone.policy is TimezonePolicy.FROM_FIELD_MAPPING:
            field = contract.timezone.field or ""
            return (
                f"{field}={record.payload.get(field)!r} нет в справочнике "
                f"{contract.timezone.mapping!r}, source default не утверждён"
            )
        return f"зону не определить по политике {contract.timezone.policy}"

    # ------------------------------------------------------------------ #
    # Разбор времени
    # ------------------------------------------------------------------ #

    def _parse(self, value: Any, spec: Any, zone_name: str) -> datetime | None:
        """Разобрать значение строго по контракту (§12 п.1)."""
        if value is None:
            return None

        kind = spec.kind
        try:
            if kind is TimestampKind.EPOCH_MILLIS:
                if not isinstance(value, int) or isinstance(value, bool):
                    return None
                return EPOCH + timedelta(milliseconds=value)

            if kind is TimestampKind.ISO_UTC:
                if not isinstance(value, str):
                    return None
                moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if moment.tzinfo is None:
                    return None
                return moment.astimezone(UTC)

            if not isinstance(value, str):
                return None

            if kind is TimestampKind.DATE_ISO:
                local = datetime.combine(date.fromisoformat(value), datetime.min.time())
            else:  # NAIVE_LOCAL_PATTERN
                local = datetime.strptime(value, spec.pattern)
                if local.tzinfo is not None:
                    return None
        except (ValueError, OverflowError):
            return None

        return _localize(local, zone_name)

    def _is_late(self, record: TimedRecord, contract: SourceContract) -> bool:
        """§33.12: запись пришла позже допустимой задержки контракта.

        Считается только там, где источник сообщает processing time. §12.2
        разрешает использовать его именно для этого — и только для этого.
        """
        if record.processing_time_utc is None or record.timestamp_utc is None:
            return False
        delay = record.processing_time_utc - record.timestamp_utc
        return delay > timedelta(hours=contract.max_delay_hours)


def _localize(local: datetime, zone_name: str) -> datetime:
    """Наивное локальное время → instant.

    `fold=0` задан явно. При переходе Казахстана на UTC+05 (1 марта 2024)
    часы сдвинулись назад, и один локальный час встречается дважды. Без
    зафиксированного правила выбор между двумя instant зависел бы от
    реализации, и golden-vectors не сошлись бы (§29.2).
    """
    zone = ZoneInfo(zone_name)
    return local.replace(tzinfo=zone, fold=0).astimezone(UTC)
