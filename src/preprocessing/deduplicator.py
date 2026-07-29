"""Deduplicator — §9.

Три разных случая, которые легко перепутать и нельзя лечить одинаково:

- **точный дубль** (§9.1) — та же запись приехала дважды: совпадают источник,
  `source_record_id` и версия payload. Остаётся одна;
- **бизнес-дубль** (§9.2) — один и тот же факт под разными `source_record_id`.
  Ловится версионируемым отпечатком из объявленных полей;
- **конфликт** (§9.3) — тот же ключ, разный payload. Здесь запрещено главное
  искушение: `drop_duplicates(keep="first")`. «Первая» запись — это та, что
  раньше легла в файл, а не та, что вернее.

Конфликт разрешается только объявленным правилом. Источник, объявивший
`update_rules: versioned_by_field`, разрешает его версией. Источник без такой
политики отправляет **всю группу** в карантин: оставить произвольного
представителя — это и есть keep-first под другим именем.

Компонент собирает поток целиком, а не обрабатывает партицию за партицией:
дубль может лежать в другом месяце, и увидеть его можно только на полном
наборе. Поэтому дедупликация — точка схождения воркеров, а не параллельный
шаг. Порядок выдачи задан каноническим положением выжившей записи
(партиция, строка), а не порядком, в котором воркеры прислали свои куски.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core.debug_dump import DebugDump, Stage
from .core.hashing import business_fingerprint, encode_timestamp
from .core.monitor import DataQualityMonitor, Metric, Total
from .core.quarantine import Quarantine, ReasonCode
from .records import TimedRecord
from .schema.source_contract import ColumnType, SourceContract, SourceContractRegistry

COMPONENT = "deduplicator"

# Имена в отпечатке, которых нет в payload: они появились по дороге.
DERIVED_CLIENT_ID = "client_id"
DERIVED_EVENT_TIME = "event_time"
DERIVED_FIELDS = frozenset({DERIVED_CLIENT_ID, DERIVED_EVENT_TIME})

# Маркеры присутствия значения в пре-образе отпечатка. Без них отсутствующее
# поле и поле с пустой строкой дали бы один отпечаток.
PRESENT = "1"
ABSENT = "0"


class DedupPolicyError(RuntimeError):
    """Ошибка политики дедупликации — блокирующая."""


class SourceDedupPolicy(BaseModel):
    """Политика одного источника."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    business_fingerprint: tuple[str, ...] | None
    """`None` — бизнес-дедупликация выключена осознанно. Ключ обязателен:
    отсутствие настройки не должно читаться как выключено."""

    @model_validator(mode="after")
    def _fingerprint_is_usable(self) -> "SourceDedupPolicy":
        if self.business_fingerprint is None:
            return self
        if len(self.business_fingerprint) < 2:
            raise ValueError(
                "отпечаток из одного поля схлопнул бы разные факты; "
                "выключайте бизнес-дедупликацию через null"
            )
        if len(set(self.business_fingerprint)) != len(self.business_fingerprint):
            raise ValueError("в отпечатке повторяются поля")
        return self


class DedupPolicy(BaseModel):
    """Политика дедупликации (§9), версионируется целиком."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dedup_policy_version: str = Field(min_length=1)
    sources: dict[str, SourceDedupPolicy] = Field(min_length=1)

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_dedup_policy(path: Path, registry: SourceContractRegistry) -> DedupPolicy:
    """Загрузить политику и проверить её против Source Contracts."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise DedupPolicyError(f"{path}: ожидался YAML-объект")
    policy = DedupPolicy.model_validate(document)

    missing = sorted(set(registry.sources) - set(policy.sources))
    if missing:
        raise DedupPolicyError(
            "нет политики дедупликации для источников: " + ", ".join(missing)
        )
    extra = sorted(set(policy.sources) - set(registry.sources))
    if extra:
        raise DedupPolicyError("политика описывает несуществующие источники: " + ", ".join(extra))

    for name in sorted(policy.sources):
        fingerprint = policy.sources[name].business_fingerprint
        if fingerprint is None:
            continue
        contract = registry.contract(name)
        for column in fingerprint:
            if column in DERIVED_FIELDS:
                continue
            spec = contract.columns.get(column)
            if spec is None:
                raise DedupPolicyError(
                    f"{name}: в отпечатке поле {column!r}, которого нет в схеме источника"
                )
            if spec.type is ColumnType.STRING_LIST:
                raise DedupPolicyError(
                    f"{name}: поле {column!r} — список; порядок элементов сделал бы "
                    "отпечаток неустойчивым"
                )
        if DERIVED_CLIENT_ID in fingerprint and contract.client_field is None:
            raise DedupPolicyError(
                f"{name}: в отпечатке client_id, но у источника нет поля клиента"
            )

    return policy


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


@dataclass
class DedupReport:
    """Что и почему было убрано."""

    exact_removed: Counter[str] = field(default_factory=Counter)
    business_removed: Counter[str] = field(default_factory=Counter)
    conflicts_resolved: Counter[str] = field(default_factory=Counter)
    conflicts_quarantined: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "DedupReport") -> None:
        self.exact_removed.update(other.exact_removed)
        self.business_removed.update(other.business_removed)
        self.conflicts_resolved.update(other.conflicts_resolved)
        self.conflicts_quarantined.update(other.conflicts_quarantined)

    def summary(self) -> dict[str, Any]:
        return {
            "exact_removed": dict(sorted(self.exact_removed.items())),
            "business_removed": dict(sorted(self.business_removed.items())),
            "conflicts_resolved_by_version": dict(sorted(self.conflicts_resolved.items())),
            "conflicts_quarantined": dict(sorted(self.conflicts_quarantined.items())),
            "removed_total": sum(self.exact_removed.values())
            + sum(self.business_removed.values()),
        }


# --------------------------------------------------------------------------- #
# Компонент
# --------------------------------------------------------------------------- #


class Deduplicator:
    """Схлопывание дублей по §9."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        policy: DedupPolicy,
        *,
        monitor: DataQualityMonitor,
        quarantine: Quarantine,
        debug: DebugDump | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self._monitor = monitor
        self._quarantine = quarantine
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = DedupReport()

    def deduplicate(self, records: Iterable[TimedRecord]) -> Iterator[TimedRecord]:
        tracing = self._debug.enabled

        collected: list[TimedRecord] = []
        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])
            collected.append(record)

        survivors = self._business_pass(self._exact_pass(collected))

        # Порядок выдачи — каноническое положение записи, а не порядок,
        # в котором её принесли воркеры (§29 п.3).
        for record in sorted(survivors, key=_canonical_position):
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [record.debug_row()])
            yield record

    # ------------------------------------------------------------------ #
    # §9.1 и §9.3 — один ключ источника
    # ------------------------------------------------------------------ #

    def _exact_pass(self, records: list[TimedRecord]) -> list[TimedRecord]:
        groups: defaultdict[tuple[str, str], list[TimedRecord]] = defaultdict(list)
        for record in records:
            groups[(record.source, record.source_record_id)].append(record)

        survivors: list[TimedRecord] = []
        for key in sorted(groups):
            self._monitor.add_total(Total.DUPLICATE_GROUPS)
            group = groups[key]
            if len(group) == 1:
                survivors.append(group[0])
                continue

            contract = self.registry.contract(group[0].source)
            kept = self._resolve_group(contract, group)
            if kept is not None:
                survivors.append(kept)
        return survivors

    def _resolve_group(
        self, contract: SourceContract, group: list[TimedRecord]
    ) -> TimedRecord | None:
        """Разрешить группу записей с одинаковым ключом источника."""
        version_field = contract.correction_reversal.version_field
        variants = {_payload_signature(record, version_field): record for record in group}

        if len(variants) == 1:
            # §9.1: совпадают источник, record_id и версия payload — и сам
            # payload тоже. Это одна и та же запись, приехавшая дважды.
            self.report.exact_removed[contract.name] += len(group) - 1
            return min(group, key=_canonical_position)

        # §9.3: ключ один, payload разный.
        self._monitor.count(Metric.DEDUP_CONFLICT_RATE)
        resolved = self._resolve_by_version(contract, group, version_field)
        if resolved is not None:
            self.report.conflicts_resolved[contract.name] += 1
            self.report.exact_removed[contract.name] += len(group) - 1
            return resolved

        # Политики нет или она не различила записи. Оставить произвольную —
        # это keep-first, который §9.3 запрещает, поэтому в карантин уходит
        # вся группа: какая из версий верна, неизвестно.
        self.report.conflicts_quarantined[contract.name] += 1
        for record in sorted(group, key=_canonical_position):
            self._quarantine.add(
                ReasonCode.UNRESOLVED_DUPLICATE_CONFLICT,
                source=record.source,
                raw_reference=record.raw_reference,
                partition=record.partition,
                detail=self._conflict_detail(contract, group, version_field),
                # Конфликт — свойство группы, и группа посчитана выше.
                # Иначе группа из трёх записей подняла бы метрику трижды,
                # а знаменатель у неё — число групп.
                count_metric=False,
            )
        return None

    def _resolve_by_version(
        self,
        contract: SourceContract,
        group: list[TimedRecord],
        version_field: str | None,
    ) -> TimedRecord | None:
        """§9.3: выбрать запись по объявленной версии.

        Возвращает `None`, если политики нет или максимум версии не един —
        тогда правило не выбрало, и выбирать больше нечем.
        """
        if version_field is None:
            return None

        versions = [(_version_of(record, version_field), record) for record in group]
        if any(version is None for version, _ in versions):
            return None

        top = max(version for version, _ in versions)
        winners = [record for version, record in versions if version == top]
        if len(winners) > 1 and len({_payload_signature(r, None) for r in winners}) > 1:
            return None
        return min(winners, key=_canonical_position)

    @staticmethod
    def _conflict_detail(
        contract: SourceContract, group: list[TimedRecord], version_field: str | None
    ) -> str:
        if version_field is None:
            return (
                f"{len(group)} записи с одним ключом и разным payload; "
                f"у источника нет correction policy (update_rules={contract.update_rules})"
            )
        seen = sorted(str(record.payload.get(version_field)) for record in group)
        return (
            f"{len(group)} записи с одним ключом и разным payload; "
            f"{version_field} не различил их: {', '.join(seen)}"
        )

    # ------------------------------------------------------------------ #
    # §9.2 — один факт под разными ключами
    # ------------------------------------------------------------------ #

    def _business_pass(self, records: list[TimedRecord]) -> list[TimedRecord]:
        groups: defaultdict[tuple[str, str], list[TimedRecord]] = defaultdict(list)
        passthrough: list[TimedRecord] = []

        for record in records:
            fingerprint = self._fingerprint(record)
            if fingerprint is None:
                passthrough.append(record)
                continue
            groups[(record.source, fingerprint)].append(record)

        survivors = passthrough
        for key in sorted(groups):
            group = groups[key]
            if len(group) > 1:
                self.report.business_removed[group[0].source] += len(group) - 1
            # Выживает запись с наименьшим source_record_id — объявленное
            # правило, а не «та, что попалась первой».
            survivors.append(min(group, key=lambda item: item.source_record_id))
        return survivors

    def _fingerprint(self, record: TimedRecord) -> str | None:
        """Версионируемый бизнес-отпечаток записи (§9.2).

        `None` — бизнес-дедупликация для источника выключена.
        """
        fields = self.policy.sources[record.source].business_fingerprint
        if fields is None:
            return None

        parts: list[str] = [self.policy.dedup_policy_version, record.source]
        for name in fields:
            value = _fingerprint_value(record, name)
            parts.extend([name, ABSENT, ""] if value is None else [name, PRESENT, value])
        return business_fingerprint(parts)


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #


def _canonical_position(record: TimedRecord) -> tuple[str, int]:
    """Каноническое место записи: партиция и строка (§29 п.1)."""
    return (record.partition, record.line_number)


def _payload_signature(record: TimedRecord, version_field: str | None) -> str:
    """Устойчивое представление payload для сравнения «тот же/не тот же».

    Ключи сортируются, поэтому порядок полей в исходном JSON на сравнение
    не влияет.
    """
    items = sorted(record.payload.items())
    if version_field is not None:
        items = [(key, value) for key, value in items if key != version_field]
    return repr(items)


def _version_of(record: TimedRecord, version_field: str) -> int | None:
    value = record.payload.get(version_field)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _fingerprint_value(record: TimedRecord, name: str) -> str | None:
    if name == DERIVED_CLIENT_ID:
        return record.client_id
    if name == DERIVED_EVENT_TIME:
        # Время в пре-образе — целое число микросекунд эпохи (§29.1 п.10),
        # а не строка ISO: формат строки зависит от локали и точности.
        return encode_timestamp(record.timestamp_utc) if record.timestamp_utc else None

    value = record.payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return None
