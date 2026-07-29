"""Bucketizer — §19, §19.1–§19.5.

Числовая бакетизация целиком принадлежит препроцессингу: токенайзер получает
готовую метку `bucket_17` как обычную категорию и границ больше не видит
(§19.5). Здесь число превращается в метку — и обратно уже не превращается.

**BUILD и ENCODE — разные вещи, и разделены они не соглашением, а типами.**
`fit_bucket_edges` считает границы по TRAIN-выборке и возвращает замороженный
артефакт. `Bucketizer` принимает артефакт в конструкторе и метода `fit` не
имеет вовсе — посчитать границы на Validation, Test или на одном примере
(§27) им физически нечем.

Границы хранятся **строками десятичных дробей**, а не числами с плавающей
точкой. Причина прикладная: значение ровно на границе бакета должно попадать
в один и тот же бакет всегда, а `0.1 + 0.2` в двоичной дроби этого не
гарантирует. §29.1 п.4 фиксирует формат float для случаев, когда float
используется; здесь он не используется, и канонический JSON получает строку,
которую любая реализация прочтёт одинаково.

Про clipping (§19.4) стоит сказать отдельно. Значение вне TRAIN-диапазона
попадает в крайний бакет **само**: сравнение с границами так и работает, и
новый бакет создать неоткуда. Смысл §19.4 не в том, чтобы это устроить, а в
том, чтобы это заметить, — поэтому здесь считаются `numeric_clip_rate` и
отдельные low/high счётчики (§19.5).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .category_normalizer import _replace_fields
from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .core.quantiles import equal_width_edges, internal_edges, quantile_edges
from .feature_projection import ProjectedRecord
from .schema.constants import MISSING
from .schema.feature_schema import FeatureSchema, FieldType

COMPONENT = "bucketizer"
LABEL_PREFIX = "bucket_"


class BucketizationError(RuntimeError):
    """Ошибка бакетизации — блокирующая."""


class BucketMethod(StrEnum):
    """Метод расчёта границ (§19.2).

    Перечислены только реализованные. `log_space` и `business_defined` из
    §19.2 сюда не внесены намеренно: объявить метод, которого нет, значит
    получить сообщение об ошибке на BUILD вместо отказа при загрузке конфига.
    """

    QUANTILE = "quantile"
    EQUAL_WIDTH = "equal_width"


class FieldBucketRule(BaseModel):
    """Правило для одного поля."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: BucketMethod
    bucket_count: int = Field(ge=2)
    """Сколько бакетов запрошено. Фактическое число может оказаться меньше:
    совпавшие границы удаляются (§19.3)."""


class BucketizationConfig(BaseModel):
    """Версионируемая конфигурация бакетизации (§19.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket_edges_version: str = Field(min_length=1)
    defaults: FieldBucketRule
    fields: dict[str, FieldBucketRule] = Field(default_factory=dict)

    def rule_for(self, name: str) -> FieldBucketRule:
        return self.fields.get(name, self.defaults)

    def state(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_bucketization_config(path: Path, schema: FeatureSchema) -> BucketizationConfig:
    """Загрузить конфиг и сверить его с Feature Schema."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise BucketizationError(f"{path}: ожидался YAML-объект")
    config = BucketizationConfig.model_validate(document)

    bucket_fields = {
        name for name, spec in schema.field_specs().items() if spec.type is FieldType.BUCKET
    }
    unknown = sorted(set(config.fields) - bucket_fields)
    if unknown:
        raise BucketizationError(
            "конфиг описывает поля, не являющиеся bucket-полями: " + ", ".join(unknown)
        )
    return config


# --------------------------------------------------------------------------- #
# Артефакт BUILD
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FieldEdges:
    """Границы и метаданные одного bucket-поля (§19.5)."""

    name: str
    method: BucketMethod
    requested_count: int
    edges: tuple[Decimal, ...]
    """Внутренние границы, по возрастанию. Бакетов ровно `len(edges) + 1`."""

    min_train: Decimal
    max_train: Decimal
    sample_size: int

    @property
    def bucket_count(self) -> int:
        """Фактическое число бакетов — §19.3 фиксирует именно его."""
        return len(self.edges) + 1

    def labels(self) -> tuple[str, ...]:
        return tuple(f"{LABEL_PREFIX}{index}" for index in range(self.bucket_count))

    def domain(self) -> tuple[str, ...]:
        """Closed-set domain поля: все метки плюс `MISSING` (§2 п.5).

        Публикуется целиком независимо от того, сколько записей попало в
        каждый бакет: §11.1 требует, чтобы токенайзер включил их все в словарь
        и не применял к ним `min_count`.
        """
        return self.labels() + (MISSING,)

    def bucket_of(self, value: Decimal) -> int:
        """Индекс бакета. Интервалы левозамкнутые: `[edge[i-1], edge[i])`."""
        return bisect.bisect_right(self.edges, value)

    def intervals(self) -> tuple[dict[str, str | None], ...]:
        """decode interval metadata (§19.5, §2 п.6).

        Крайние интервалы открыты: значение ниже `min_train` всё равно
        попадает в нулевой бакет (§19.4), и закрывать его нижней границей
        значило бы обещать, что таких значений не бывает.
        """
        bounds: list[dict[str, str | None]] = []
        for index in range(self.bucket_count):
            low = str(self.edges[index - 1]) if index > 0 else None
            high = str(self.edges[index]) if index < len(self.edges) else None
            bounds.append({"label": f"{LABEL_PREFIX}{index}", "low": low, "high": high})
        return tuple(bounds)

    def as_state(self) -> dict[str, Any]:
        """JSON-совместимый вид для §29.1 и §30.

        Границы — строки: точное десятичное представление читается одинаково
        любой реализацией, а двоичная дробь сдвинула бы значение, стоящее
        ровно на границе, в соседний бакет.
        """
        return {
            "method": str(self.method),
            "requested_bucket_count": self.requested_count,
            "bucket_count": self.bucket_count,
            "edges": [str(edge) for edge in self.edges],
            "min_train_edge": str(self.min_train),
            "max_train_edge": str(self.max_train),
            "sample_size": self.sample_size,
            "labels": list(self.labels()),
            "domain": list(self.domain()),
            "missing_token": MISSING,
            "intervals": [dict(item) for item in self.intervals()],
        }


@dataclass(frozen=True)
class BucketEdges:
    """Замороженный артефакт BUILD: границы всех bucket-полей."""

    version: str
    fields: dict[str, FieldEdges]

    def bucket_field_domains(self) -> dict[str, tuple[str, ...]]:
        """Пункт 5 контракта §2 — то, что уходит токенайзеру как whitelist."""
        return {name: self.fields[name].domain() for name in sorted(self.fields)}

    def bucket_metadata(self) -> dict[str, Any]:
        """Пункт 6 контракта §2 — интервалы только для decode и observability."""
        return {
            name: {
                "method": str(self.fields[name].method),
                "bucket_count": self.fields[name].bucket_count,
                "intervals": [dict(item) for item in self.fields[name].intervals()],
            }
            for name in sorted(self.fields)
        }

    def state(self) -> dict[str, Any]:
        """Состояние для §29.1 и §30."""
        return {
            "bucket_edges_version": self.version,
            "fields": {name: self.fields[name].as_state() for name in sorted(self.fields)},
        }


# --------------------------------------------------------------------------- #
# BUILD — расчёт границ
# --------------------------------------------------------------------------- #


def fit_bucket_edges(
    sample: dict[str, Sequence[Decimal]],
    config: BucketizationConfig,
    schema: FeatureSchema,
) -> BucketEdges:
    """Посчитать границы по TRAIN-выборке (§19).

    `sample` — выход `DeterministicSampler`: значения уже прошли §17 и §18,
    `MISSING` и невалидные в нём отсутствуют (§19.1).
    """
    bucket_fields = sorted(
        name for name, spec in schema.field_specs().items() if spec.type is FieldType.BUCKET
    )

    missing = sorted(set(bucket_fields) - set(sample))
    if missing:
        raise BucketizationError("в выборке нет значений для полей: " + ", ".join(missing))

    fields: dict[str, FieldEdges] = {}
    for name in bucket_fields:
        values = sorted(sample[name])
        if not values:
            # Поле без единого валидного значения бакетизировать нечем.
            # Молча выдать один бакет — спрятать сломанный источник.
            raise BucketizationError(
                f"{name}: в TRAIN-выборке нет ни одного валидного значения (§19.1)"
            )
        fields[name] = _fit_field(name, values, config.rule_for(name))

    return BucketEdges(version=config.bucket_edges_version, fields=fields)


def _fit_field(name: str, values: list[Decimal], rule: FieldBucketRule) -> FieldEdges:
    if rule.method is BucketMethod.QUANTILE:
        raw = quantile_edges(values, rule.bucket_count)
    else:
        raw = equal_width_edges(values, rule.bucket_count)

    edges = internal_edges(raw, values)
    return FieldEdges(
        name=name,
        method=rule.method,
        requested_count=rule.bucket_count,
        edges=edges,
        min_train=values[0],
        max_train=values[-1],
        sample_size=len(values),
    )


# --------------------------------------------------------------------------- #
# ENCODE — применение границ
# --------------------------------------------------------------------------- #


@dataclass
class BucketizerReport:
    """Сколько значений размечено и сколько прижато к краям (§19.5)."""

    assigned: dict[str, int] = field(default_factory=dict)
    clipped_low: dict[str, int] = field(default_factory=dict)
    clipped_high: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "BucketizerReport") -> None:
        for source, target in (
            (other.assigned, self.assigned),
            (other.clipped_low, self.clipped_low),
            (other.clipped_high, self.clipped_high),
            (other.missing, self.missing),
        ):
            for name, count in source.items():
                target[name] = target.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "assigned_by_field": dict(sorted(self.assigned.items())),
            "clipped_low_by_field": dict(sorted(self.clipped_low.items())),
            "clipped_high_by_field": dict(sorted(self.clipped_high.items())),
            "missing_by_field": dict(sorted(self.missing.items())),
        }


class Bucketizer:
    """Применение замороженных границ (§19, ENCODE).

    Метода `fit` здесь нет: §19 разрешает считать границы только на TRAIN,
    и посчитать их этим классом нечем.
    """

    def __init__(
        self,
        edges: BucketEdges,
        schema: FeatureSchema,
        *,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.edges = edges
        self.schema = schema
        self._domains = edges.bucket_field_domains()
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = BucketizerReport()

    def transform(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            result = self._transform_one(record)
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _transform_one(self, record: ProjectedRecord) -> ProjectedRecord:
        if not record.fields:
            return record

        updated: dict[str, Any] = {}
        for name, value in record.fields.items():
            field_edges = self.edges.fields.get(name)
            if field_edges is None:
                updated[name] = value
                continue
            updated[name] = self._label_of(field_edges, value)
        return _replace_fields(record, updated)

    def _label_of(self, field_edges: FieldEdges, value: Any) -> str:
        name = field_edges.name
        if value == MISSING:
            # §19.5: кроме меток бакетов поле может содержать только MISSING.
            self.report.missing[name] = self.report.missing.get(name, 0) + 1
            return MISSING

        if not isinstance(value, Decimal):
            raise BucketizationError(
                f"{name}: на бакетизацию пришло значение типа {type(value).__name__}; "
                "компонент работает после §17 и §18"
            )

        self._monitor.add_total(Total.BUCKET_VALUES)

        # §19.4: значение вне TRAIN-диапазона попадает в крайний бакет само.
        # Считаем это отдельно — сам факт выхода за диапазон и есть то, что
        # §19.5 требует наблюдать.
        if value < field_edges.min_train:
            self.report.clipped_low[name] = self.report.clipped_low.get(name, 0) + 1
            self._monitor.count(Metric.NUMERIC_CLIP_RATE)
        elif value > field_edges.max_train:
            self.report.clipped_high[name] = self.report.clipped_high.get(name, 0) + 1
            self._monitor.count(Metric.NUMERIC_CLIP_RATE)

        label = f"{LABEL_PREFIX}{field_edges.bucket_of(value)}"
        self._check_domain(name, label)
        self.report.assigned[name] = self.report.assigned.get(name, 0) + 1
        return label

    def _check_domain(self, name: str, label: str) -> None:
        """§33.4: метка вне опубликованного domain — critical.

        По построению такого быть не может; сработать это способно только
        если границы и domain приехали из разных BUILD. Тогда токенайзер
        получит значение, которого нет в его словаре, и связь между
        артефактами уже потеряна — продолжать нельзя.
        """
        if label not in self._domains[name]:
            self._monitor.count(Metric.BUCKET_DOMAIN_VIOLATION_RATE)
            raise BucketizationError(
                f"{name}: метка {label} вне опубликованного domain — "
                "границы и bucket_field_domains из разных BUILD (§33.4)"
            )
