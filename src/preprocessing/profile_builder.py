"""ProfileBuilder — §6, §24.

Собирает один профиль на клиента: снимок на момент T плюс life-long признаки.

**Инвариант порядконезависимости.** Все шесть признаков §24 — агрегаты:
счётчики и минимумы по множеству событий ≤ T. Порядок им не нужен, и это не
везение, а условие, при котором расчёт вообще можно ставить сюда: сортировка
timeline (§13, §26) идёт **позже** по цепочке, и признак, которому нужен
порядок, получил бы здесь порядок поступления записей — то есть порядок
партиций, зависящий от нарезки потока.

Правило закреплено интерфейсом: аккумулятор видит события **по одному**
(`add`) и последовательности не получает никогда. Написать «тип третьего
события» через него неудобно, а перестановочная проверка ловит попытку.
Признак, которому нужен порядок, обязан жить после TimelineBuilder.

**Снимок выбирается прошлым, а не ближайшим** (§6 п.1). Будущих снимков здесь
уже нет — их отсекла §14 раньше по цепочке, — поэтому «взять последний из
доступных» и «взять последний до T» это одно и то же действие, и ошибиться
нечем.

**Клиент без снимка** получает профиль с обязательными полями `MISSING` и
флагом (§6 п.3). Не первый будущий снимок: его в потоке нет и взять его
неоткуда.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .feature_projection import PROFILE_SECTION, ProjectedRecord
from .numeric_validator import InvalidReason, validate_numeric
from .records import QualityFlag
from .schema.constants import MISSING
from .schema.feature_schema import FeatureSchema, FieldType
from .schema.source_contract import SourceContractRegistry

COMPONENT = "profile_builder"

# Имена life-long признаков — перечень §24. Порядок фиксирован ради
# читаемости отчёта; на результат он не влияет.
LIFELONG_FIELDS: tuple[str, ...] = (
    "account_age_bucket",
    "first_seen_age_bucket",
    "first_topup_age_bucket",
    "lifetime_event_count_bucket",
    "lifetime_transaction_count_bucket",
    "lifetime_product_count_bucket",
)


class ProfilePolicyError(RuntimeError):
    """Ошибка политики профиля — блокирующая."""


class TopupPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_types: tuple[str, ...] = Field(min_length=1)
    direction_field: str = Field(min_length=1)
    direction_value: str = Field(min_length=1)


class ProfilePolicy(BaseModel):
    """Версионируемая политика профиля (§6, §24)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_policy_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    snapshot_date_field: str = Field(min_length=1)
    account_open_field: str = Field(min_length=1)
    products_field: str = Field(min_length=1)
    transaction_event_types: tuple[str, ...] = Field(min_length=1)
    topup: TopupPolicy

    def state(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_profile_policy(
    path: Path, registry: SourceContractRegistry, schema: FeatureSchema, event_types: Iterable[str]
) -> ProfilePolicy:
    """Загрузить политику и сверить её с контрактом, схемой и типами событий."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ProfilePolicyError(f"{path}: ожидался YAML-объект")
    policy = ProfilePolicy.model_validate(document)

    contract = registry.contract(policy.source)
    for label, column in (
        ("snapshot_date_field", policy.snapshot_date_field),
        ("account_open_field", policy.account_open_field),
        ("products_field", policy.products_field),
    ):
        if column not in contract.columns:
            raise ProfilePolicyError(f"{policy.source}: {label}={column!r} нет в схеме источника")

    approved = set(event_types)
    unknown = sorted(
        (set(policy.transaction_event_types) | set(policy.topup.event_types)) - approved
    )
    if unknown:
        raise ProfilePolicyError(
            "политика ссылается на типы событий вне §10: " + ", ".join(unknown)
        )

    profile_specs = schema.section_specs(PROFILE_SECTION)
    missing = sorted(set(LIFELONG_FIELDS) - set(profile_specs))
    if missing:
        raise ProfilePolicyError(
            "life-long признаки §24 не объявлены в Feature Schema: " + ", ".join(missing)
        )
    not_computed = sorted(name for name in LIFELONG_FIELDS if not profile_specs[name].computed)
    if not_computed:
        raise ProfilePolicyError(
            "life-long признаки обязаны быть computed — их некому прочитать из "
            "снимка: " + ", ".join(not_computed)
        )

    return policy


# --------------------------------------------------------------------------- #
# Аккумуляторы life-long признаков
# --------------------------------------------------------------------------- #


class LifelongAggregate:
    """Агрегат по множеству событий клиента (§24).

    Видит события по одному и никогда — последовательность. Это и есть
    закреплённое правило: результат не должен зависеть от порядка, потому что
    порядка на этом шаге ещё нет.
    """

    def add(self, record: ProjectedRecord) -> None:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def value(self, cutoff: datetime) -> Decimal | str:  # pragma: no cover - интерфейс
        raise NotImplementedError


class EventCount(LifelongAggregate):
    """`lifetime_event_count` — сколько событий у клиента до T."""

    def __init__(self, event_types: frozenset[str] | None = None) -> None:
        self._types = event_types
        self._count = 0

    def add(self, record: ProjectedRecord) -> None:
        if self._types is None or record.event_type in self._types:
            self._count += 1

    def value(self, cutoff: datetime) -> Decimal | str:
        return Decimal(self._count)


class EarliestMoment(LifelongAggregate):
    """Возраст самого раннего подходящего события в днях на T.

    Минимум — операция коммутативная и ассоциативная, поэтому порядок
    поступления на результат не влияет.
    """

    def __init__(self, predicate) -> None:
        self._predicate = predicate
        self._earliest: datetime | None = None

    def add(self, record: ProjectedRecord) -> None:
        if record.timestamp_utc is None or not self._predicate(record):
            return
        if self._earliest is None or record.timestamp_utc < self._earliest:
            self._earliest = record.timestamp_utc

    def value(self, cutoff: datetime) -> Decimal | str:
        if self._earliest is None:
            return MISSING
        return Decimal((cutoff - self._earliest).days)


@dataclass
class ProfileReport:
    """Сколько профилей собрано и чего в них не хватило."""

    clients: int = 0
    without_snapshot: int = 0
    lifelong_missing: dict[str, int] = field(default_factory=dict)
    lifelong_invalid: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "ProfileReport") -> None:
        self.clients += other.clients
        self.without_snapshot += other.without_snapshot
        for source, target in (
            (other.lifelong_missing, self.lifelong_missing),
            (other.lifelong_invalid, self.lifelong_invalid),
        ):
            for name, count in source.items():
                target[name] = target.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "clients": self.clients,
            "without_snapshot": self.without_snapshot,
            "lifelong_missing_by_field": dict(sorted(self.lifelong_missing.items())),
            "lifelong_invalid_by_field": dict(sorted(self.lifelong_invalid.items())),
        }


class ProfileBuilder:
    """Профиль клиента на момент T (§6) с life-long признаками (§24)."""

    def __init__(
        self,
        schema: FeatureSchema,
        policy: ProfilePolicy,
        *,
        cutoff: datetime,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self.policy = policy
        self.cutoff = cutoff
        self._specs = schema.section_specs(PROFILE_SECTION)
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = ProfileReport()

    def build(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        """Пропустить события и заменить снимки одним профилем на клиента."""
        tracing = self._debug.enabled

        events: list[ProjectedRecord] = []
        snapshots: dict[str, list[ProjectedRecord]] = {}
        aggregates: dict[str, dict[str, LifelongAggregate]] = {}

        for record in records:
            if record.source == self.policy.source:
                if tracing:
                    self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])
                if record.client_id:
                    snapshots.setdefault(record.client_id, []).append(record)
                continue

            events.append(record)
            if record.client_id is None or record.event_type is None:
                continue
            self._accumulators(aggregates, record.client_id)
            for aggregate in aggregates[record.client_id].values():
                aggregate.add(record)

        yield from events

        for client_id in sorted(set(snapshots) | set(aggregates)):
            profile = self._profile_of(
                client_id, snapshots.get(client_id, []), aggregates.get(client_id, {})
            )
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [profile.debug_row()])
            yield profile

    def _accumulators(
        self, store: dict[str, dict[str, LifelongAggregate]], client_id: str
    ) -> None:
        if client_id in store:
            return
        transactions = frozenset(self.policy.transaction_event_types)
        topup_types = frozenset(self.policy.topup.event_types)
        store[client_id] = {
            "lifetime_event_count_bucket": EventCount(),
            "lifetime_transaction_count_bucket": EventCount(transactions),
            "first_seen_age_bucket": EarliestMoment(lambda record: True),
            "first_topup_age_bucket": EarliestMoment(
                lambda record: record.event_type in topup_types
                and record.fields.get(self.policy.topup.direction_field)
                == self.policy.topup.direction_value
            ),
        }

    # ------------------------------------------------------------------ #
    # Сборка профиля
    # ------------------------------------------------------------------ #

    def _profile_of(
        self,
        client_id: str,
        snapshots: list[ProjectedRecord],
        aggregates: dict[str, LifelongAggregate],
    ) -> ProjectedRecord:
        self.report.clients += 1
        self._monitor.add_total(Total.CLIENTS_PROCESSED)

        chosen = self._latest_snapshot(snapshots)
        fields: dict[str, Any] = dict(chosen.fields) if chosen is not None else {}
        flags = chosen.quality_flags if chosen is not None else ()

        if chosen is None:
            # §6 п.3: снимка нет — обязательные поля MISSING и флаг.
            # Будущий снимок не берётся: его в потоке нет.
            self.report.without_snapshot += 1
            self._monitor.count(Metric.PROFILE_MISSING_RATE)
            flags = (QualityFlag.PROFILE_SNAPSHOT_MISSING,)
            fields = {
                name: MISSING
                for name, spec in self._specs.items()
                if spec.required and not spec.computed
            }

        for name in LIFELONG_FIELDS:
            fields[name] = self._lifelong_value(name, chosen, aggregates)

        return ProjectedRecord(
            source=self.policy.source,
            partition=chosen.partition if chosen is not None else "",
            line_number=chosen.line_number if chosen is not None else 0,
            source_record_id=chosen.source_record_id if chosen is not None else client_id,
            source_schema_version=chosen.source_schema_version if chosen is not None else "",
            client_ref=chosen.client_ref if chosen is not None else None,
            payload=chosen.payload if chosen is not None else {},
            client_id=client_id,
            timestamp_utc=chosen.timestamp_utc if chosen is not None else self.cutoff,
            calendar_timezone=chosen.calendar_timezone if chosen is not None else None,
            processing_time_utc=None,
            quality_flags=flags,
            event_type=None,
            event_id=None,
            fields=fields,
            schema_section=PROFILE_SECTION,
        )

    def _latest_snapshot(self, snapshots: list[ProjectedRecord]) -> ProjectedRecord | None:
        """§6 п.1: последний снимок не позже T.

        Будущие снимки отсечены §14 раньше по цепочке, поэтому «последний из
        доступных» и «последний до T» здесь совпадают. Tie-break по
        `source_record_id` — два снимка одной датой не должны решаться
        порядком партиций.
        """
        if not snapshots:
            return None
        return max(
            snapshots,
            key=lambda record: (record.timestamp_utc, record.source_record_id),
        )

    def _lifelong_value(
        self,
        name: str,
        snapshot: ProjectedRecord | None,
        aggregates: dict[str, LifelongAggregate],
    ) -> Any:
        raw = self._raw_lifelong(name, snapshot, aggregates)
        if raw == MISSING:
            self.report.lifelong_missing[name] = self.report.lifelong_missing.get(name, 0) + 1
            return MISSING

        # §24: raw lifetime value → validation → bucketization.
        # Проверка — общая с §17, не своя: два места с одним правилом уже
        # дважды оборачивались расхождением.
        spec = self._specs[name]
        assert spec.numeric is not None
        self._monitor.add_total(Total.NUMERIC_VALUES)
        checked = validate_numeric(raw, spec.numeric)
        if isinstance(checked, InvalidReason):
            self._monitor.count(Metric.NUMERIC_BUSINESS_RANGE_ERROR_RATE, label=name)
            self._monitor.count(Metric.MISSING_RATE, label=name)
            self.report.lifelong_invalid[name] = self.report.lifelong_invalid.get(name, 0) + 1
            return MISSING
        return checked

    def _raw_lifelong(
        self,
        name: str,
        snapshot: ProjectedRecord | None,
        aggregates: dict[str, LifelongAggregate],
    ) -> Any:
        if name == "account_age_bucket":
            return self._account_age(snapshot)
        if name == "lifetime_product_count_bucket":
            return self._product_count(snapshot)

        aggregate = aggregates.get(name)
        if aggregate is None:
            # Клиент без единого события: считать нечего, и ноль здесь был бы
            # неправдой для возраста первого события.
            return Decimal(0) if name.startswith("lifetime_") else MISSING
        return aggregate.value(self.cutoff)

    def _account_age(self, snapshot: ProjectedRecord | None) -> Any:
        if snapshot is None:
            return MISSING
        opened = snapshot.payload.get(self.policy.account_open_field)
        if not isinstance(opened, str):
            return MISSING
        try:
            opened_date = date.fromisoformat(opened)
        except ValueError:
            return MISSING
        return Decimal((self.cutoff.date() - opened_date).days)

    def _product_count(self, snapshot: ProjectedRecord | None) -> Any:
        if snapshot is None:
            return MISSING
        products = snapshot.payload.get(self.policy.products_field)
        if not isinstance(products, list):
            return MISSING
        return Decimal(len(products))


def lifelong_field_names(schema: FeatureSchema) -> tuple[str, ...]:
    """Life-long поля схемы — те, что заполняет этот компонент."""
    specs = schema.section_specs(PROFILE_SECTION)
    return tuple(
        name
        for name in sorted(specs)
        if specs[name].computed and specs[name].type is FieldType.BUCKET
    )
