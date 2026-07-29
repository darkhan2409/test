"""Data-quality метрики — §33.

Метрики регламента — это доли, а не абсолютные счётчики: «> 0.1% — warning»
бессмысленно без знаменателя. Поэтому монитор считает две вещи отдельно:
числитель (сколько раз случилось) и знаменатель (сколько раз могло случиться),
а доля выводится в отчёте.

Пороги §33 здесь **не применяются**: регламент говорит «уточняются по baseline
банка», а на синтетике никакого baseline нет. Значения считаем и печатаем,
решение о тревоге — за пределами этого этапа. Пороговые формулировки хранятся
рядом со спецификацией метрики, чтобы не искать их в документе.

Кардинальность лейблов: только ограниченные множества — имя поля, имя
источника, регион. `client_id`, `event_id` и произвольный текст в лейбл
не берём никогда.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Total(StrEnum):
    """Знаменатели: сколько раз операция вообще выполнялась."""

    RECORDS_READ = "records_read"
    RECORDS_MAPPED = "records_mapped"
    EVENTS_VALIDATED = "events_validated"
    TIMESTAMPS_PARSED = "timestamps_parsed"
    TIMEZONES_RESOLVED = "timezones_resolved"
    NUMERIC_VALUES = "numeric_values"
    FX_CONVERSIONS = "fx_conversions"
    BUCKET_VALUES = "bucket_values"
    DUPLICATE_GROUPS = "duplicate_groups"
    SESSIONS_BUILT = "sessions_built"
    CLIENTS_PROCESSED = "clients_processed"
    FIELDS_EMITTED = "fields_emitted"
    QUARANTINED = "quarantined"


class Metric(StrEnum):
    """Числители — по одному на подраздел §33."""

    CUTOFF_VIOLATION_RATE = "cutoff_violation_rate"
    SCHEMA_VIOLATION_RATE = "schema_violation_rate"
    UNKNOWN_EVENT_TYPE_RATE = "unknown_event_type_rate"
    BUCKET_DOMAIN_VIOLATION_RATE = "bucket_domain_violation_rate"
    DEDUP_CONFLICT_RATE = "dedup_conflict_rate"
    MISSING_RATE = "missing_rate"
    NUMERIC_PARSE_ERROR_RATE = "numeric_parse_error_rate"
    NUMERIC_NAN_RATE = "numeric_nan_rate"
    NUMERIC_BUSINESS_RANGE_ERROR_RATE = "numeric_business_range_error_rate"
    NUMERIC_CLIP_RATE = "numeric_clip_rate"
    FX_MISSING_RATE = "fx_missing_rate"
    TIMESTAMP_ERROR_RATE = "timestamp_error_rate"
    CALENDAR_TIMEZONE_FALLBACK_RATE = "calendar_timezone_fallback_rate"
    LATE_ARRIVING_RATE = "late_arriving_rate"
    SESSION_ANOMALY_RATE = "session_anomaly_rate"
    PROFILE_MISSING_RATE = "profile_missing_rate"


@dataclass(frozen=True)
class MetricSpec:
    """Чем метрика делится, откуда она в регламенте и что считается тревогой."""

    denominator: Total
    spec_section: str
    thresholds: str
    label: str | None = None  # имя измерения, если метрика считается в разрезе


METRIC_SPECS: dict[Metric, MetricSpec] = {
    Metric.CUTOFF_VIOLATION_RATE: MetricSpec(
        Total.EVENTS_VALIDATED, "§33.1", "любое значение > 0 — critical"
    ),
    Metric.SCHEMA_VIOLATION_RATE: MetricSpec(
        Total.EVENTS_VALIDATED, "§33.2", "> 0.1% — warning; > 1% — critical"
    ),
    Metric.UNKNOWN_EVENT_TYPE_RATE: MetricSpec(
        Total.RECORDS_MAPPED, "§33.3", "> 0% — warning; > 0.1% — critical"
    ),
    Metric.BUCKET_DOMAIN_VIOLATION_RATE: MetricSpec(
        Total.BUCKET_VALUES, "§33.4", "любое значение > 0 — critical"
    ),
    Metric.DEDUP_CONFLICT_RATE: MetricSpec(
        Total.DUPLICATE_GROUPS, "§33.5", "рост > 2x baseline — warning; > 5x — critical"
    ),
    Metric.MISSING_RATE: MetricSpec(
        Total.FIELDS_EMITTED, "§33.6", "рост > 2x — warning; > 5x — critical", label="field"
    ),
    Metric.NUMERIC_PARSE_ERROR_RATE: MetricSpec(
        Total.NUMERIC_VALUES, "§33.7", "<= 0.01% — normal; > 0.01% — warning; > 0.1% — critical"
    ),
    # §17.2 требует мониторить три метрики числовых полей, а §33 определяет
    # только первую. Две ниже заведены по §17.2 без порогов: придумывать
    # пороги за регламент нельзя, а не считать величину, которую он требует
    # считать, — тем более.
    Metric.NUMERIC_NAN_RATE: MetricSpec(
        Total.NUMERIC_VALUES, "§17.2", "порог в §33 не определён"
    ),
    Metric.NUMERIC_BUSINESS_RANGE_ERROR_RATE: MetricSpec(
        Total.NUMERIC_VALUES, "§17.2", "порог в §33 не определён", label="field"
    ),
    Metric.NUMERIC_CLIP_RATE: MetricSpec(
        Total.BUCKET_VALUES, "§33.8", "рост > 2x — warning; > 5x — critical"
    ),
    Metric.FX_MISSING_RATE: MetricSpec(
        Total.FX_CONVERSIONS, "§33.9", "<= 0.1% — normal; > 0.1% — warning; > 1% — critical"
    ),
    Metric.TIMESTAMP_ERROR_RATE: MetricSpec(
        Total.TIMESTAMPS_PARSED, "§33.10", "> 0% — warning; > 0.1% — critical"
    ),
    Metric.CALENDAR_TIMEZONE_FALLBACK_RATE: MetricSpec(
        Total.TIMEZONES_RESOLVED,
        "§33.11",
        "контролируется по source/region; резкий рост — ухудшение timezone metadata",
        label="source",
    ),
    Metric.LATE_ARRIVING_RATE: MetricSpec(
        Total.RECORDS_READ, "§33.12", "сравнивается с baseline по источнику", label="source"
    ),
    Metric.SESSION_ANOMALY_RATE: MetricSpec(
        Total.SESSIONS_BUILT,
        "§33.13",
        "negative/excessive duration, empty session, too many values",
    ),
    Metric.PROFILE_MISSING_RATE: MetricSpec(
        Total.CLIENTS_PROCESSED, "§33.14", "контролируется по сегментам клиентов"
    ),
}


class DataQualityMonitor:
    """Счётчики прогона.

    Складывается из воркеров через `merge`: сложение счётчиков коммутативно,
    поэтому итог не зависит от порядка слияния (§29 п.3).
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._labelled: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._totals: Counter[str] = Counter()
        self._label_totals: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def count(self, metric: Metric, *, label: str | None = None, by: int = 1) -> None:
        """Зафиксировать событие метрики (числитель)."""
        self._counts[str(metric)] += by
        if label is not None:
            self._labelled[str(metric)][label] += by

    def add_total(self, total: Total, *, by: int = 1) -> None:
        """Зафиксировать выполненную операцию (знаменатель)."""
        self._totals[str(total)] += by

    def add_label_total(self, metric: Metric, label: str, *, by: int = 1) -> None:
        """Знаменатель для метрики в разрезе — например, сколько раз поле
        вообще эмитилось, чтобы посчитать его missing_rate."""
        self._label_totals[str(metric)][label] += by

    def merge(self, other: "DataQualityMonitor") -> None:
        self._counts.update(other._counts)
        self._totals.update(other._totals)
        for metric, values in other._labelled.items():
            self._labelled[metric].update(values)
        for metric, values in other._label_totals.items():
            self._label_totals[metric].update(values)

    def report(self) -> dict[str, Any]:
        """Отчёт прогона: значения, знаменатели и доли.

        Метрики без единого срабатывания тоже попадают в отчёт — «ноль» и
        «не считали» это разные вещи, и отличать их важно.
        """
        return {
            "totals": {name: self._totals[name] for name in sorted(map(str, Total))},
            "metrics": {
                str(metric): self._metric_report(metric) for metric in sorted(map(str, Metric))
            },
        }

    def _metric_report(self, metric_name: str) -> dict[str, Any]:
        metric = Metric(metric_name)
        spec = METRIC_SPECS[metric]
        count = self._counts[metric_name]
        total = self._totals[str(spec.denominator)]

        report: dict[str, Any] = {
            "spec_section": spec.spec_section,
            "thresholds": spec.thresholds,
            "count": count,
            "denominator": str(spec.denominator),
            "denominator_value": total,
            "rate": _rate(count, total),
        }

        if spec.label is not None:
            report["label"] = spec.label
            report["by_label"] = {
                label: {
                    "count": value,
                    "denominator_value": self._label_totals[metric_name][label],
                    "rate": _rate(value, self._label_totals[metric_name][label]),
                }
                for label, value in sorted(self._labelled[metric_name].items())
            }
        return report


def _rate(count: int, total: int) -> float | None:
    """Доля или `None`, если операция ни разу не выполнялась.

    Ноль вместо `None` соврал бы: «0% ошибок» и «ни одной попытки» — разные
    состояния, и второе обычно означает, что компонент не отработал.
    """
    if total <= 0:
        return None
    return count / total
