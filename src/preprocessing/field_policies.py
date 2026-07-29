"""FieldPolicies — §21, §22, §23.

Последний фильтр перед тем, как поля уйдут к токенайзеру. Три правила, и все
три — про то, чего в модели быть **не должно**.

**§21 многозначные поля.** Порядок либо значим, либо нет, и это решение
человека, а не источника: `order_significant` обязателен у каждого
многозначного поля. Значим — сохраняем хронологию и режем хвост; незначим —
сортируем и режем. Лимит берётся из Feature Schema (§11), где он и объявлен.

**Обрезкой владеет только этот компонент.** Sessionizer лимит знает, но
применяет его лишь для аномалии §33.13: одно правило — одно место.
`screens_count` при этом уже посчитан по полному списку, потому что §19
бакетизирует именно его, а не длину усечённого.

**§22 высокая кардинальность.** `merchant_id`, `atm_id` и прочие технические
идентификаторы уходят здесь. `RARE` препроцессинг не назначает — §22 прямо
это запрещает; открытые множества уходят токенайзеру как есть.

**§23 текст.** Свободного текста тут нет и быть не может: в `FieldType` нет
текстового типа, объявить такое поле нечем. Что остаётся проверить — что
через `source_field` в модель не утекает колонка, помеченная в Source
Contract как прямой идентификатор. Это проверяется при загрузке, до данных.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .category_normalizer import _replace_fields
from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor
from .feature_projection import ProjectedRecord
from .schema.constants import MISSING
from .schema.feature_schema import (
    FeatureSchema,
    FieldType,
    HighCardinalityPolicy,
    SharedFieldSpec,
)
from .schema.source_contract import PiiClass, SourceContractRegistry, SourceKind

COMPONENT = "field_policies"

TEXT_POLICY_VERSION = "1.0.0"

TEXT_POLICY: dict[str, Any] = {
    "text_policy_version": TEXT_POLICY_VERSION,
    # §23: свободного текста в модели нет. Формулировка «forbidden» здесь не
    # обещание кода себя вести — объявить текстовое поле нечем, см. ниже.
    "free_text_fields": "forbidden",
    # Перечень берётся из самого `FieldType`, а не переписывается сюда руками.
    # Это и есть смысл записи: запрет §23 выражен отсутствием текстового типа,
    # и если такой тип однажды появится, состояние изменится само — а вместе с
    # ним и `preprocessing_state_sha256`. Копия списка молчала бы.
    "declarable_field_types": sorted(str(item) for item in FieldType),
    # §23: колонка, помеченная владельцем источника как прямой идентификатор,
    # не может быть `source_field` признака. Проверяется до данных.
    "direct_identifier_as_source_field": "forbidden",
    "high_cardinality_strategies": sorted(str(item) for item in HighCardinalityPolicy),
    # §22: препроцессинг `RARE` не назначает — это решение токенайзера после
    # подсчёта частот по TRAIN.
    "rare_assignment": "tokenizer_only",
}
"""Text policy (§23) и её соседи из §22 — пункт перечней §30 и §31.

Отдельного конфига у неё нет и не должно быть: политика состоит из запретов,
а запреты здесь выражены устройством типов. Конфиг, которым запрет можно
ослабить, был бы ровно тем механизмом, которого §23 не предполагает.
"""


class FieldPolicyError(RuntimeError):
    """Нарушение политики полей — блокирующая ошибка."""


def check_field_policies(
    schema: FeatureSchema, registry: SourceContractRegistry, *, default_max_values: int
) -> None:
    """Проверить политики полей до обработки данных (§21, §22, §23).

    Всё, что можно поймать на схеме и контрактах, ловится здесь: ошибка в
    политике полей — это ошибка конфигурации, и обнаруживать её на середине
    прогона незачем.
    """
    specs = schema.field_specs()

    for name in sorted(specs):
        spec = specs[name]

        if spec.multivalue and spec.max_values_per_field is None and default_max_values <= 0:
            raise FieldPolicyError(f"{name}: нет лимита значений ни в схеме, ни в умолчании (§21)")

        # §22: технический ID, объявленный excluded, не должен одновременно
        # считаться входом модели — иначе непонятно, что победит.
        if spec.high_cardinality is HighCardinalityPolicy.EXCLUDE and spec.model_input:
            raise FieldPolicyError(
                f"{name}: high_cardinality=exclude вместе с model_input=true (§22)"
            )

    _check_no_direct_identifiers(schema, registry)


def _check_no_direct_identifiers(
    schema: FeatureSchema, registry: SourceContractRegistry
) -> None:
    """§23: PII не доходит до токенайзера.

    Проверяется по классификации Source Contract, а не по имени поля:
    «выглядит как телефон» — не критерий, а `pii: direct_identifier` —
    решение владельца источника.
    """
    columns: dict[str, PiiClass] = {}
    for contract in registry.sources.values():
        for column, spec in contract.columns.items():
            known = columns.get(column)
            # Одноимённые колонки разных источников с разным классом PII —
            # берём строгий: поле уйдёт в модель одно.
            if known is None or spec.pii is PiiClass.DIRECT_IDENTIFIER:
                columns[column] = spec.pii

    leaking: list[str] = []
    for section in list(schema.events) + ["PROFILE"]:
        for name, spec in schema.section_specs(section).items():
            if spec.computed or not spec.model_input:
                continue
            if columns.get(spec.source_field) is PiiClass.DIRECT_IDENTIFIER:
                leaking.append(f"{section}.{name} ← {spec.source_field}")

    if leaking:
        raise FieldPolicyError(
            "прямые идентификаторы доходят до модели (§23): " + ", ".join(sorted(leaking))
        )


@dataclass
class FieldPolicyReport:
    """Что убрано и что обрезано."""

    excluded: dict[str, int] = field(default_factory=dict)
    truncated: dict[str, int] = field(default_factory=dict)
    reordered: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "FieldPolicyReport") -> None:
        for source, target in (
            (other.excluded, self.excluded),
            (other.truncated, self.truncated),
            (other.reordered, self.reordered),
        ):
            for name, count in source.items():
                target[name] = target.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "excluded_by_field": dict(sorted(self.excluded.items())),
            "truncated_by_field": dict(sorted(self.truncated.items())),
            "sorted_by_field": dict(sorted(self.reordered.items())),
        }


class FieldPolicies:
    """Применение §21, §22 и §23 к готовым полям."""

    def __init__(
        self,
        schema: FeatureSchema,
        *,
        default_max_values: int,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self.default_max_values = default_max_values
        self._specs = schema.field_specs()
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = FieldPolicyReport()

    def apply(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            result = self._apply_one(record)
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _apply_one(self, record: ProjectedRecord) -> ProjectedRecord:
        if not record.fields:
            return record

        updated: dict[str, Any] = {}
        for name, value in record.fields.items():
            spec = self._specs.get(name)
            if spec is None:
                updated[name] = value
                continue

            if not spec.model_input:
                # §22: технический ID до модели не доходит. Убирается здесь,
                # а не пропускается при проекции, — чтобы решение было видно
                # в дампе как отдельный шаг, а не как отсутствие строки.
                self.report.excluded[name] = self.report.excluded.get(name, 0) + 1
                continue

            updated[name] = self._limit(spec, value) if spec.multivalue else value

        return _replace_fields(record, updated)

    def _limit(self, spec: SharedFieldSpec, value: Any) -> Any:
        """§21 пп. 1–5: порядок и обрезка многозначного поля."""
        if value == MISSING or not isinstance(value, list):
            return value

        values = list(value)
        if not spec.order_significant:
            # §21 п.2: порядок незначим — сортируем, чтобы одно и то же
            # множество всегда давало одну последовательность токенов.
            ordered = sorted(values)
            if ordered != values:
                self.report.reordered[spec.name] = self.report.reordered.get(spec.name, 0) + 1
            values = ordered

        limit = spec.max_values_per_field or self.default_max_values
        if len(values) > limit:
            # §21 пп. 4–5: при значимом порядке хронология и есть утверждённый
            # priority — обрезается хвост, а не произвольные элементы.
            self.report.truncated[spec.name] = self.report.truncated.get(spec.name, 0) + 1
            values = values[:limit]

        return values


def model_fields(schema: FeatureSchema) -> tuple[str, ...]:
    """Поля, доходящие до токенайзера, — то, что осталось после §22."""
    return tuple(
        name for name, spec in schema.field_specs().items() if spec.model_input
    )
