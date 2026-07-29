"""CategoryNormalizer — §16, §16.1, §16.2, §16.3.

Приводит категориальные значения к каноническому виду. Порядок операций
фиксирован: trim → Unicode NFKC → регистр → alias mapping. Порядок не
косметика: alias-таблица ищется уже по приведённому значению, иначе в неё
пришлось бы вносить `kzt`, `KZT`, `Kzt` и `ТЕНГЕ` отдельными строками.

Что компонент делать **не** имеет права.

**Заводить новую категорию** (§16.1). У closed-set поля значение вне domain
даёт либо `MISSING`, либо schema violation — что именно, решает конфиг. Тихо
расширить domain нельзя: токенайзер включает его в словарь целиком, и
незаявленное значение сломало бы соответствие словаря и данных.

**Решать судьбу редких значений** (§16.2). Open-set поле нормализуется и
остаётся как есть. `RARE` и `[UNK]` — работа токенайзера после заморозки
словаря; здесь про частоты не известно ничего и знать не положено.

**Пользоваться `OTHER` как свалкой** (§16.3). `OTHER` допустим только как
утверждённая категория со стабильным смыслом, поэтому здесь он никуда не
подставляется: ошибка нормализации даёт `MISSING`, а не `OTHER`.

`MISSING` через компонент проходит нетронутым: он поставлен §15 осознанно,
и нормализовать его нечего.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric
from .core.quarantine import Quarantine, ReasonCode
from .feature_projection import ProjectedRecord
from .schema.constants import MISSING
from .schema.feature_schema import FeatureSchema, FieldType, VocabularyPolicy

COMPONENT = "category_normalizer"


class CategoryMappingError(RuntimeError):
    """Ошибка справочника категорий — блокирующая."""


class CaseRule(StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    NONE = "none"


class UnicodeRule(StrEnum):
    NFKC = "NFKC"
    NFC = "NFC"
    NONE = "none"


class ViolationPolicy(StrEnum):
    """Что делать со значением вне closed-set domain (§16.1).

    Третьего варианта нет намеренно: «создать новую категорию» регламент
    запрещает, и запрет выражен отсутствием возможности его объявить.
    """

    MISSING = "missing"
    SCHEMA_VIOLATION = "schema_violation"


class NormalizationDefaults(BaseModel):
    """Правила, действующие на все категориальные поля."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trim: bool = True
    unicode: UnicodeRule = UnicodeRule.NFKC
    case: CaseRule = CaseRule.UPPER
    on_violation: ViolationPolicy = ViolationPolicy.MISSING


class FieldCategoryRule(BaseModel):
    """Справочник одного поля."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aliases: dict[str, str] = Field(default_factory=dict)
    on_violation: ViolationPolicy | None = None

    @model_validator(mode="after")
    def _aliases_are_canonical(self) -> "FieldCategoryRule":
        for source, target in self.aliases.items():
            if not source or not target:
                raise ValueError("пустое значение в alias-таблице")
            if target in self.aliases and self.aliases[target] != target:
                # Иначе результат зависел бы от числа проходов по таблице.
                raise ValueError(
                    f"цепочка алиасов: {source!r} → {target!r} → {self.aliases[target]!r}; "
                    "таблица обязана быть плоской"
                )
        return self


class CategoryMapping(BaseModel):
    """Версионируемый справочник нормализации (§16)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category_mapping_version: str = Field(min_length=1)
    defaults: NormalizationDefaults = Field(default_factory=NormalizationDefaults)
    fields: dict[str, FieldCategoryRule] = Field(default_factory=dict)

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_category_mapping(path: Path, schema: FeatureSchema) -> CategoryMapping:
    """Загрузить справочник и сверить его с Feature Schema."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise CategoryMappingError(f"{path}: ожидался YAML-объект")
    mapping = CategoryMapping.model_validate(document)

    specs = schema.field_specs()
    unknown = sorted(set(mapping.fields) - set(specs))
    if unknown:
        raise CategoryMappingError(
            "справочник описывает поля вне Feature Schema: " + ", ".join(unknown)
        )

    categorical = {
        name
        for name, spec in specs.items()
        if spec.type is FieldType.CATEGORICAL
        and spec.vocabulary_policy is not VocabularyPolicy.EXCLUDED
    }
    uncovered = sorted(categorical - set(mapping.fields))
    if uncovered:
        # Правило по умолчанию есть, но молчание в конфиге читалось бы как
        # «поле не категориальное». Требуем явную строку — хотя бы с пустой
        # alias-таблицей.
        raise CategoryMappingError(
            "категориальные поля не описаны в справочнике: " + ", ".join(uncovered)
        )

    # Алиас, ведущий в значение вне domain, — тихая порча closed-set поля.
    for name in sorted(mapping.fields):
        spec = specs[name]
        if spec.vocabulary_policy is not VocabularyPolicy.CLOSED_SET or not spec.domain:
            continue
        outside = sorted(
            {target for target in mapping.fields[name].aliases.values() if target not in spec.domain}
        )
        if outside:
            raise CategoryMappingError(
                f"{name}: алиасы ведут в значения вне domain: {', '.join(outside)} (§16.1)"
            )

    return mapping


@dataclass
class NormalizationReport:
    """Что изменилось и что не сошлось."""

    normalized: dict[str, int] = field(default_factory=dict)
    aliased: dict[str, int] = field(default_factory=dict)
    violations: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "NormalizationReport") -> None:
        for source, target in (
            (other.normalized, self.normalized),
            (other.aliased, self.aliased),
            (other.violations, self.violations),
        ):
            for name, count in source.items():
                target[name] = target.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "values_changed_by_field": dict(sorted(self.normalized.items())),
            "alias_hits_by_field": dict(sorted(self.aliased.items())),
            "closed_set_violations_by_field": dict(sorted(self.violations.items())),
        }


class CategoryNormalizer:
    """Канонизация категориальных значений."""

    def __init__(
        self,
        schema: FeatureSchema,
        mapping: CategoryMapping,
        *,
        monitor: DataQualityMonitor,
        quarantine: Quarantine,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self.mapping = mapping
        self._specs = schema.field_specs()
        self._monitor = monitor
        self._quarantine = quarantine
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = NormalizationReport()

    def normalize(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            result = self._normalize_one(record)
            if result is None:
                continue
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _normalize_one(self, record: ProjectedRecord) -> ProjectedRecord | None:
        if not record.fields:
            return record

        updated: dict[str, Any] = {}
        for name, value in record.fields.items():
            spec = self._specs.get(name)
            if spec is None or spec.type is not FieldType.CATEGORICAL:
                updated[name] = value
                continue
            if spec.vocabulary_policy is VocabularyPolicy.EXCLUDED:
                # Поле не идёт в модель (§22): нормализовать его — работа
                # впустую, а тронутое значение выглядело бы утверждённым.
                updated[name] = value
                continue

            outcome = self._normalize_field(spec.name, value, record)
            if outcome is _REJECT:
                return None
            updated[name] = outcome

        return _replace_fields(record, updated)

    def _normalize_field(self, name: str, value: Any, record: ProjectedRecord) -> Any:
        if isinstance(value, list):
            normalized = [self._normalize_scalar(name, item, record) for item in value]
            if any(item is _REJECT for item in normalized):
                return _REJECT
            # Пропуски внутри многозначного поля не превращают всё поле в
            # MISSING: остальные значения остаются осмысленными.
            kept = [item for item in normalized if item != MISSING]
            return kept if kept else MISSING
        return self._normalize_scalar(name, value, record)

    def _normalize_scalar(self, name: str, value: Any, record: ProjectedRecord) -> Any:
        if value == MISSING:
            # Пропуск поставлен §15 осознанно — нормализовать нечего.
            return value
        if not isinstance(value, str):
            return value

        rule = self.mapping.fields.get(name, FieldCategoryRule())
        canonical = self._canonicalize(value)
        canonical = rule.aliases.get(canonical, canonical)

        if canonical != value:
            self.report.normalized[name] = self.report.normalized.get(name, 0) + 1
        if self._canonicalize(value) in rule.aliases:
            self.report.aliased[name] = self.report.aliased.get(name, 0) + 1

        spec = self._specs[name]
        if spec.vocabulary_policy is VocabularyPolicy.CLOSED_SET and spec.domain:
            if canonical not in spec.domain:
                return self._on_violation(name, value, canonical, rule, record)

        return canonical

    def _canonicalize(self, value: str) -> str:
        """trim → Unicode → регистр. Порядок фиксирован (§16)."""
        defaults = self.mapping.defaults
        text = value.strip() if defaults.trim else value
        if defaults.unicode is not UnicodeRule.NONE:
            text = unicodedata.normalize(str(defaults.unicode), text)
        if defaults.case is CaseRule.UPPER:
            text = text.upper()
        elif defaults.case is CaseRule.LOWER:
            text = text.lower()
        return text

    def _on_violation(
        self,
        name: str,
        original: str,
        canonical: str,
        rule: FieldCategoryRule,
        record: ProjectedRecord,
    ) -> Any:
        """§16.1: значение вне закрытого набора."""
        self.report.violations[name] = self.report.violations.get(name, 0) + 1

        policy = rule.on_violation or self.mapping.defaults.on_violation
        if policy is ViolationPolicy.MISSING:
            # Метрика именно MISSING: значение не потеряно как запись, оно
            # стало пропуском конкретного поля.
            self._monitor.count(Metric.MISSING_RATE, label=name)
            return MISSING

        self._monitor.count(Metric.SCHEMA_VIOLATION_RATE)
        self._quarantine.add(
            ReasonCode.SOURCE_CONTRACT_VIOLATION,
            source=record.source,
            raw_reference=record.raw_reference,
            partition=record.partition,
            detail=(
                f"{name}={original!r} (после канонизации {canonical!r}) вне closed-set domain; "
                "новая категория не создаётся (§16.1)"
            ),
            count_metric=False,
        )
        return _REJECT


class _Reject:
    """Маркер «запись уходит в карантин целиком»."""

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return "<REJECT>"


_REJECT = _Reject()


def _replace_fields(record: ProjectedRecord, fields: dict[str, Any]) -> ProjectedRecord:
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
        schema_section=record.schema_section,
    )
