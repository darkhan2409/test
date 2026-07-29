"""FXNormalizer — §18, §18.1, §18.2, §18.3.

Переводит сумму в базовую валюту по **историческому** курсу — тому, что
действовал на момент события. Не сегодняшнему: иначе транзакция 2023 года
пересчиталась бы по курсу 2026-го, и сумма в тенге зависела бы от даты
запуска пайплайна, а не от факта.

Правило поиска курса — дословно §18.2: последний доступный курс **не позднее**
события. Будущий курс запрещён, и запрет здесь структурный: индекс отдаёт
только записи с датой не больше даты события, взять более позднюю просто
неоткуда.

Про давность. Сравнение идёт в **календарных днях**, а не в часах между
инстантами. Курс публикуется на дату, и Пятница → Понедельник это три дня,
а не «три дня десять часов». По часам обычные выходные съедали бы весь бюджет
в три дня, и каждое утро понедельника давало бы MISSING.

Если допустимого курса нет — §18.2 п.5 предписывает ровно четыре вещи, и все
четыре выполняются: сумма становится `MISSING`, исходная валюта остаётся,
**событие не удаляется**, растёт `fx_missing_rate`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .category_normalizer import _replace_fields
from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .feature_projection import ProjectedRecord
from .records import TimedRecord
from .schema.constants import MISSING
from .schema.feature_schema import FeatureSchema, FieldSpec
from .schema.source_contract import SourceContractRegistry

COMPONENT = "fx_normalizer"


class FxConfigError(RuntimeError):
    """Ошибка конфигурации FX — блокирующая."""


class FxConfig(BaseModel):
    """Описание источника курсов (§18.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fx_normalization_version: str = Field(min_length=1)
    base_currency: str = Field(min_length=1)
    source: str = Field(min_length=1)
    rate_date_field: str = Field(min_length=1)
    currency_field: str = Field(min_length=1)
    rate_field: str = Field(min_length=1)
    rate_source_field: str = Field(min_length=1)
    approved_rate_sources: tuple[str, ...] = Field(min_length=1)

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_fx_config(path: Path, registry: SourceContractRegistry) -> FxConfig:
    """Загрузить конфиг и сверить его с контрактом источника курсов."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FxConfigError(f"{path}: ожидался YAML-объект")
    config = FxConfig.model_validate(document)

    contract = registry.contract(config.source)
    for label, column in (
        ("rate_date_field", config.rate_date_field),
        ("currency_field", config.currency_field),
        ("rate_field", config.rate_field),
        ("rate_source_field", config.rate_source_field),
    ):
        if column not in contract.columns:
            raise FxConfigError(f"{config.source}: {label}={column!r} нет в схеме источника")

    return config


# --------------------------------------------------------------------------- #
# Таблица курсов
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FxRateTable:
    """Курсы по валютам, отсортированные по дате.

    Хранится как два параллельных списка на валюту, чтобы поиск «последний не
    позднее» был двоичным: на 200 клиентах это неважно, на боевых объёмах
    линейный поиск по каждой сумме — заметная разница.
    """

    dates: dict[str, list[date]]
    rates: dict[str, list[Decimal]]
    version: str
    rejected: int = 0
    """Строк курса, отклонённых при построении: неутверждённый поставщик или
    неразбираемое значение. Считается, чтобы «курсов меньше, чем ожидали»
    не пришлось выяснять по журналу."""

    @classmethod
    def build(
        cls, records: Iterable[TimedRecord], config: FxConfig
    ) -> "FxRateTable":
        collected: dict[str, dict[date, Decimal]] = {}
        rejected = 0

        for record in records:
            if record.source != config.source:
                continue

            provider = record.payload.get(config.rate_source_field)
            if provider not in config.approved_rate_sources:
                # §18.1: источник курса обязан быть утверждённым.
                rejected += 1
                continue

            currency = record.payload.get(config.currency_field)
            raw_date = record.payload.get(config.rate_date_field)
            raw_rate = record.payload.get(config.rate_field)
            if not isinstance(currency, str) or not isinstance(raw_date, str):
                rejected += 1
                continue

            try:
                rate_date = date.fromisoformat(raw_date)
                rate = Decimal(str(raw_rate))
            except (ValueError, InvalidOperation):
                rejected += 1
                continue
            if not rate.is_finite() or rate <= 0:
                rejected += 1
                continue

            collected.setdefault(currency, {})[rate_date] = rate

        dates: dict[str, list[date]] = {}
        rates: dict[str, list[Decimal]] = {}
        for currency in sorted(collected):
            ordered = sorted(collected[currency])
            dates[currency] = ordered
            rates[currency] = [collected[currency][day] for day in ordered]

        return cls(
            dates=dates,
            rates=rates,
            version=config.fx_normalization_version,
            rejected=rejected,
        )

    def rate_at(self, currency: str, moment: date) -> tuple[Decimal, date] | None:
        """Последний курс не позднее даты (§18.2 п.1).

        Будущий курс вернуть нельзя: срез идёт строго левее правой границы.
        """
        days = self.dates.get(currency)
        if not days:
            return None
        position = bisect.bisect_right(days, moment)
        if position == 0:
            return None
        return self.rates[currency][position - 1], days[position - 1]

    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted(self.dates))


# --------------------------------------------------------------------------- #
# Компонент
# --------------------------------------------------------------------------- #


@dataclass
class FxReport:
    """Что пересчиталось и что нет."""

    converted: int = 0
    base_currency_skipped: int = 0
    exact_rate: int = 0
    fallback_rate: int = 0
    missing_by_reason: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "FxReport") -> None:
        self.converted += other.converted
        self.base_currency_skipped += other.base_currency_skipped
        self.exact_rate += other.exact_rate
        self.fallback_rate += other.fallback_rate
        for reason, count in other.missing_by_reason.items():
            self.missing_by_reason[reason] = self.missing_by_reason.get(reason, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "converted": self.converted,
            "base_currency_skipped": self.base_currency_skipped,
            "exact_rate": self.exact_rate,
            "fallback_rate": self.fallback_rate,
            "missing_by_reason": dict(sorted(self.missing_by_reason.items())),
        }


class FXNormalizer:
    """Пересчёт сумм в базовую валюту по историческому курсу."""

    def __init__(
        self,
        schema: FeatureSchema,
        config: FxConfig,
        table: FxRateTable,
        *,
        max_staleness: timedelta,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self.config = config
        self.table = table
        self.max_staleness_days = max_staleness.days
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = FxReport()

    def normalize(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            result = self._normalize_one(record)
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _normalize_one(self, record: ProjectedRecord) -> ProjectedRecord:
        if not record.fields or record.schema_section is None:
            return record

        specs = self.schema.section_specs(record.schema_section)
        targets = [
            name
            for name in record.fields
            if name in specs and specs[name].fx_normalized
        ]
        if not targets:
            return record

        updated = dict(record.fields)
        for name in targets:
            updated[name] = self._convert(specs[name], record, updated)
        return _replace_fields(record, updated)

    def _convert(
        self, spec: FieldSpec, record: ProjectedRecord, fields: dict[str, Any]
    ) -> Any:
        amount = fields[spec.name]
        if amount == MISSING:
            # §17 уже признал сумму невалидной — пересчитывать нечего.
            return amount

        self._monitor.add_total(Total.FX_CONVERSIONS)
        currency = fields.get(spec.currency_field or "")

        if currency == self.config.base_currency:
            # §18: базовая валюта не конвертируется.
            self.report.base_currency_skipped += 1
            return amount

        if not isinstance(currency, str) or currency == MISSING:
            # Валюта неизвестна — пересчитать нельзя, и придумывать её нельзя.
            return self._missing(spec.name, "currency_missing")

        if record.timestamp_utc is None:
            return self._missing(spec.name, "timestamp_missing")

        event_date = record.timestamp_utc.date()
        found = self.table.rate_at(currency, event_date)
        if found is None:
            return self._missing(spec.name, "no_rate_before_event")

        rate, rate_date = found
        staleness = (event_date - rate_date).days
        if staleness > self.max_staleness_days:
            # §18.2 п.5: курс есть, но он слишком старый.
            return self._missing(spec.name, "rate_too_stale")

        if staleness == 0:
            self.report.exact_rate += 1
        else:
            self.report.fallback_rate += 1
        self.report.converted += 1
        return amount * rate

    def _missing(self, name: str, reason: str) -> str:
        """§18.2 п.5: сумма становится MISSING, событие остаётся.

        Удалять событие нельзя: факт покупки состоялся независимо от того,
        удалось ли пересчитать её сумму.
        """
        self._monitor.count(Metric.FX_MISSING_RATE)
        self._monitor.count(Metric.MISSING_RATE, label=name)
        self.report.missing_by_reason[reason] = self.report.missing_by_reason.get(reason, 0) + 1
        return MISSING
