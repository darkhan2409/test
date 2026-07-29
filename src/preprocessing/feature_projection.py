"""FeatureSchemaRegistry и MISSING — §11, §11.2, §15.

Здесь запись перестаёт быть «тем, что прислал источник», и становится набором
признаков, объявленных Feature Schema. Три правила определяют результат.

**Что не в схеме — того нет.** Поле, которого нет в Feature Schema для этого
типа события, не попадает в `fields`. Не «на всякий случай оставим»: схема —
контракт с токенайзером, и всё, что мимо неё, для модели не существует.

**Применимо, но нет значения → `MISSING`. Неприменимо → поля нет вовсе**
(§15.1). Разницу задаёт `required` в схеме: `true` — поле применимо к каждой
записи этого типа, и его отсутствие даёт `MISSING`; `false` — применимо не
всегда, и тогда ключа просто нет. Регламент этой разницы не проговаривает,
трактовка выведена из §15.1.

**`MISSING` — не ноль и не `[UNK]`** (§15.3, §15.4). Ноль бывает настоящим
значением, а `[UNK]` появляется у токенайзера после заморозки словаря. Здесь
не делается ни того, ни другого: пустое значение становится строкой `MISSING`
и остаётся ею до конца препроцессинга.

Значения на этом шаге ещё сырые: в `amount_base_bucket` лежит `"15 000,50"`,
а не `bucket_17`. Имя конечное, форма — нет. Приведение значений идёт дальше
по цепочке §37.2 (§16 категории, §17 числа, §18 FX, §19 бакеты).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .event_mapper import MappedRecord
from .schema.constants import MISSING, PROFILE_SECTION
from .schema.feature_schema import EventFeatureSchema, FeatureSchema, FieldSpec
from .schema.source_contract import SourceContractRegistry, SourceKind

COMPONENT = "feature_schema"


class FeatureSchemaError(RuntimeError):
    """Ошибка Feature Schema или её стыка с контрактами — блокирующая."""


class MissingPolicy(BaseModel):
    """Что считается пропуском (§15.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placeholders: tuple[str, ...] = ()
    """Утверждённые source placeholders. Список закрыт: «похоже на заглушку»
    не критерий, иначе однажды исчезнет настоящее значение."""

    trim: bool = True

    def normalize(self, value: str) -> str:
        return value.strip() if self.trim else value

    def is_missing(self, value: Any) -> bool:
        """`NULL`, пустая строка после trim, утверждённый placeholder (§15.2)."""
        if value is None:
            return True
        if isinstance(value, str):
            text = self.normalize(value)
            if not text:
                return True
            return _fold(text) in self._folded_placeholders
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        return False

    @property
    def _folded_placeholders(self) -> frozenset[str]:
        # Сравнение без учёта регистра и Unicode-формы: `"N/A"` и `"n/a"`
        # — одна и та же заглушка, а нормализация регистра идёт только на
        # следующем шаге (§16).
        return frozenset(_fold(item) for item in self.placeholders)


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


@dataclass
class ProjectionReport:
    """Что получилось после применения схемы."""

    projected: int = 0
    missing_by_field: dict[str, int] = None  # type: ignore[assignment]
    omitted_by_field: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.missing_by_field is None:
            self.missing_by_field = {}
        if self.omitted_by_field is None:
            self.omitted_by_field = {}

    def merge(self, other: "ProjectionReport") -> None:
        self.projected += other.projected
        for name, count in other.missing_by_field.items():
            self.missing_by_field[name] = self.missing_by_field.get(name, 0) + count
        for name, count in other.omitted_by_field.items():
            self.omitted_by_field[name] = self.omitted_by_field.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "projected": self.projected,
            "missing_by_field": dict(sorted(self.missing_by_field.items())),
            "omitted_by_field": dict(sorted(self.omitted_by_field.items())),
        }


@dataclass(frozen=True)
class ProjectedRecord(MappedRecord):
    """Запись с полями строго по Feature Schema (§11) и пропусками по §15."""

    fields: dict[str, Any] = None  # type: ignore[assignment]
    """Признаки под конечными именами схемы. Значения ещё сырые: приведение
    идёт дальше по цепочке."""

    schema_section: str | None = None
    """`event_type` или `PROFILE` — какая секция схемы применялась. `None` —
    запись не описывается схемой вовсе (справочник курсов)."""

    def debug_row(self) -> dict[str, Any]:
        return {
            **super().debug_row(),
            "schema_section": self.schema_section,
            "fields": dict(self.fields or {}),
        }


def load_feature_schema(
    path: Path, registry: SourceContractRegistry, event_types: Iterable[str]
) -> tuple[FeatureSchema, MissingPolicy]:
    """Загрузить схему и MISSING policy, сверив их с контрактами и типами событий."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FeatureSchemaError(f"{path}: ожидался YAML-объект")

    policy = MissingPolicy.model_validate(document.pop("missing_policy", {}))
    schema = FeatureSchema.model_validate(document)

    approved = set(event_types)
    unknown = sorted(set(schema.events) - approved)
    if unknown:
        raise FeatureSchemaError(
            "схема описывает типы событий вне утверждённого списка §10: " + ", ".join(unknown)
        )
    uncovered = sorted(approved - set(schema.events))
    if uncovered:
        # Тип события без схемы дал бы событие без единого признака — токен
        # типа без содержимого. Это почти наверняка забытая строка конфига.
        raise FeatureSchemaError(
            "у типов событий нет полей в Feature Schema: " + ", ".join(uncovered)
        )

    _check_profile_source(schema, registry)
    return schema, policy


def _check_profile_source(schema: FeatureSchema, registry: SourceContractRegistry) -> None:
    """Профильные поля обязаны существовать в профильном источнике."""
    profile_sources = [
        name for name, contract in registry.sources.items() if contract.kind is SourceKind.PROFILE
    ]
    if not profile_sources:
        raise FeatureSchemaError("нет профильного источника, а секция PROFILE описана (§11.2)")

    columns = set(registry.contract(profile_sources[0]).columns)
    unknown = sorted(
        spec.source_field
        for spec in schema.profile.fields
        if not spec.computed and spec.source_field not in columns
    )
    if unknown:
        raise FeatureSchemaError(
            f"PROFILE: полей нет в схеме источника {profile_sources[0]}: " + ", ".join(unknown)
        )


class FeatureProjector:
    """Применение Feature Schema и MISSING policy."""

    def __init__(
        self,
        schema: FeatureSchema,
        policy: MissingPolicy,
        registry: SourceContractRegistry,
        *,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self.policy = policy
        self.registry = registry
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = ProjectionReport()

    def project(self, records: Iterable[MappedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            projected = self._project_one(record)
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [projected.debug_row()])
            yield projected

    def _project_one(self, record: MappedRecord) -> ProjectedRecord:
        section, specs = self._section_of(record)
        if specs is None:
            # Справочник курсов: схема его не описывает, и это не пропуск.
            return _projected(record, fields={}, section=None)

        fields: dict[str, Any] = {}
        for spec in specs:
            if spec.computed:
                # Значение считает более поздний компонент (§24). Поставить
                # здесь MISSING значило бы объявить пропуском то, что просто
                # ещё не посчитано.
                continue
            outcome = self._value_of(spec, record.payload)
            if outcome is _OMIT:
                self.report.omitted_by_field[spec.name] = (
                    self.report.omitted_by_field.get(spec.name, 0) + 1
                )
                continue

            fields[spec.name] = outcome
            self._monitor.add_total(Total.FIELDS_EMITTED)
            self._monitor.add_label_total(Metric.MISSING_RATE, spec.name)
            if outcome == MISSING:
                self._monitor.count(Metric.MISSING_RATE, label=spec.name)
                self.report.missing_by_field[spec.name] = (
                    self.report.missing_by_field.get(spec.name, 0) + 1
                )

        self.report.projected += 1
        return _projected(record, fields=fields, section=section)

    def _section_of(
        self, record: MappedRecord
    ) -> tuple[str | None, tuple[FieldSpec, ...] | None]:
        if record.event_type is not None:
            schema: EventFeatureSchema | None = self.schema.events.get(record.event_type)
            if schema is None:
                raise FeatureSchemaError(
                    f"{record.raw_reference}: у типа {record.event_type} нет секции Feature Schema"
                )
            return record.event_type, schema.ordered_fields()

        contract = self.registry.contract(record.source)
        if contract.kind is SourceKind.PROFILE:
            return PROFILE_SECTION, self.schema.profile.ordered_fields()
        return None, None

    def _value_of(self, spec: FieldSpec, payload: Any) -> Any:
        """Значение поля по §15.1 и §15.2."""
        if spec.source_field not in payload:
            # Ключа нет. Применимо ли поле — решает схема, а не данные:
            # иначе «нет значения» и «неприменимо» стали бы одним и тем же.
            return MISSING if spec.required else _OMIT

        value = payload[spec.source_field]
        if self.policy.is_missing(value):
            return MISSING

        if isinstance(value, str):
            return self.policy.normalize(value)
        if isinstance(value, (list, tuple)):
            return [item for item in value]
        return value


class _Omit:
    """Маркер «поле неприменимо, ключа быть не должно» (§15.1)."""

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return "<OMIT>"


_OMIT = _Omit()


def _projected(
    record: MappedRecord, *, fields: dict[str, Any], section: str | None
) -> ProjectedRecord:
    return ProjectedRecord(
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
        fields=fields,
        schema_section=section,
    )
