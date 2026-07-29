"""NumericValidator — §17, §17.1, §17.2, §17.3.

Превращает то, что источник считает числом, в число — или в `MISSING`.
Третьего исхода нет, и это главное в компоненте.

§17.2 перечисляет запреты прямо: невалидное число нельзя заменять на `0`,
отправлять в `[UNK]`, создавать под него новый бакет и нельзя падать на
обычном production inference. Все четыре запрета выполнены **устройством**,
а не проверками: единственное, что метод возврата умеет вернуть кроме числа,
— это `MISSING`. Нуля в коде нет, `[UNK]` здесь неоткуда взять, бакетов ещё
не существует, а исключений компонент не бросает.

Разбор идёт по `Decimal`, а не по `float`: деньги в двоичной дроби — способ
получить `0.1 + 0.2` в бакетных границах. Точность здесь дешёвая, а ошибка
округления всплыла бы только на сравнении хэшей.

Два разных понятия, которые легко перепутать:

- **business range** (§17.2) — значение бессмысленно по существу
  (отрицательный возраст), результат `MISSING`;
- **clipping** (§19.4) — значение осмысленно, но вне TRAIN-диапазона,
  результат — крайний бакет. Это работа §19, и здесь её нет.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator

from .category_normalizer import _replace_fields
from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .feature_projection import ProjectedRecord
from .schema.constants import MISSING
from .schema.feature_schema import (
    FeatureSchema,
    FieldSpec,
    FieldType,
    NumericSpec,
    NumericType,
    ParsingLocale,
)

COMPONENT = "numeric_validator"

# Утверждённые разделители тысяч (§17.1). Узкий и неразрывный пробелы сюда
# входят потому, что выгрузки из офисных инструментов приносят именно их,
# а на вид они неотличимы от обычного пробела.
THOUSANDS_SEPARATORS = (" ", " ", " ", " ", "'")

# Строки, которые Python разобрал бы как число, а §17.2 объявляет невалидными.
NON_FINITE = frozenset({"nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity",
                        "-infinity", "+infinity"})


class InvalidReason(str):
    """Причина невалидности — только для отчёта и метрик."""


PARSE_ERROR = InvalidReason("parse_error")
NON_FINITE_VALUE = InvalidReason("non_finite")
OUT_OF_RANGE = InvalidReason("business_range")


@dataclass
class NumericReport:
    """Что не разобралось и почему."""

    parsed: int = 0
    parse_errors: dict[str, int] = field(default_factory=dict)
    non_finite: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "NumericReport") -> None:
        self.parsed += other.parsed
        for source, target in (
            (other.parse_errors, self.parse_errors),
            (other.non_finite, self.non_finite),
            (other.out_of_range, self.out_of_range),
        ):
            for name, count in source.items():
                target[name] = target.get(name, 0) + count

    def summary(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "parse_errors_by_field": dict(sorted(self.parse_errors.items())),
            "non_finite_by_field": dict(sorted(self.non_finite.items())),
            "out_of_business_range_by_field": dict(sorted(self.out_of_range.items())),
        }


def _parse_value(value: Any, numeric: NumericSpec) -> Decimal | InvalidReason:
    """§17.1: подготовка строки и преобразование."""
    if isinstance(value, bool):
        # bool — подкласс int, и `True` стал бы суммой в один тенге.
        return PARSE_ERROR
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # Float до сюда доходить не должен: источник присылает строку или
        # целое, а Decimal из float тащит двоичную погрешность.
        return PARSE_ERROR
    if not isinstance(value, str):
        return PARSE_ERROR

    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return PARSE_ERROR
    if text.lower().lstrip("+-") in NON_FINITE or text.lower() in NON_FINITE:
        return NON_FINITE_VALUE

    if numeric.locale is ParsingLocale.RU_KZ:
        for separator in THOUSANDS_SEPARATORS:
            text = text.replace(separator, "")
        if "," in text and "." in text:
            # Обе формы сразу — угадывать, что из них дробное, нельзя.
            return PARSE_ERROR
        text = text.replace(",", ".")
    else:
        if "," in text:
            return PARSE_ERROR

    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return PARSE_ERROR

    if not parsed.is_finite():
        return NON_FINITE_VALUE
    return parsed


def validate_numeric(value: Any, numeric: NumericSpec) -> Decimal | InvalidReason:
    """Разобрать и проверить значение по §17. Единственная реализация правил.

    Вынесено из класса затем, чтобы у §17 остался один владелец. Life-long
    признаки (§24) рождаются числами и разбора не требуют, но проверку
    диапазона проходить обязаны — §24 прямо ставит `validation` перед
    `bucketization`. Собственная проверка внутри ProfileBuilder означала бы
    два места с одним правилом, а это уже дважды оборачивалось расхождением.
    """
    parsed = _parse_value(value, numeric)
    if isinstance(parsed, InvalidReason):
        return parsed

    # `numeric_type` описывает форму **исходного** значения, поэтому
    # целочисленность проверяется до применения `scale`: сумма в тиынах
    # целая, а те же деньги в тенге — уже нет.
    if numeric.numeric_type is NumericType.INTEGER and parsed != parsed.to_integral_value():
        # Дробная часть у целочисленного поля — не «округлим», а «не то
        # значение»: округление здесь молча меняло бы данные.
        return PARSE_ERROR

    scaled = parsed * numeric.scale
    if not numeric.signed and scaled < 0:
        # §17.3: правило знака задано полю, а не типу.
        return OUT_OF_RANGE
    if numeric.min_value is not None and scaled < numeric.min_value:
        return OUT_OF_RANGE
    if numeric.max_value is not None and scaled > numeric.max_value:
        return OUT_OF_RANGE
    return scaled


class NumericValidator:
    """Разбор и проверка числовых полей."""

    def __init__(
        self,
        schema: FeatureSchema,
        *,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        self.schema = schema
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = NumericReport()

    def validate(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            result = self._validate_one(record)
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [result.debug_row()])
            yield result

    def _validate_one(self, record: ProjectedRecord) -> ProjectedRecord:
        if not record.fields or record.schema_section is None:
            return record

        # Спеки берутся по секции, а не по имени поля: у `amount_base_bucket`
        # в карточном процессинге тиыны и `scale: 0.01`, а в ядре платежей —
        # строка в тенге с запятой. Общий по имени спек разобрал бы одно из
        # двух неверно.
        specs = self.schema.section_specs(record.schema_section)

        updated: dict[str, Any] = {}
        for name, value in record.fields.items():
            spec = specs.get(name)
            if spec is None or spec.type is not FieldType.BUCKET:
                updated[name] = value
                continue
            updated[name] = self._value_of(spec, value)
        return _replace_fields(record, updated)

    def _value_of(self, spec: FieldSpec, value: Any) -> Any:
        if value == MISSING:
            # Пропуск поставлен §15 — разбирать нечего.
            return value

        self._monitor.add_total(Total.NUMERIC_VALUES)
        numeric = spec.numeric
        assert numeric is not None  # гарантировано валидатором Feature Schema

        result = validate_numeric(value, numeric)
        if isinstance(result, InvalidReason):
            return self._invalid(spec.name, result)

        self.report.parsed += 1
        return result

    def _invalid(self, name: str, reason: InvalidReason) -> str:
        """§17.2: единственный исход невалидного числа — `MISSING`."""
        if reason is PARSE_ERROR:
            self._monitor.count(Metric.NUMERIC_PARSE_ERROR_RATE)
            self.report.parse_errors[name] = self.report.parse_errors.get(name, 0) + 1
        elif reason is NON_FINITE_VALUE:
            self._monitor.count(Metric.NUMERIC_NAN_RATE)
            self.report.non_finite[name] = self.report.non_finite.get(name, 0) + 1
        else:
            self._monitor.count(Metric.NUMERIC_BUSINESS_RANGE_ERROR_RATE, label=name)
            self.report.out_of_range[name] = self.report.out_of_range.get(name, 0) + 1

        self._monitor.count(Metric.MISSING_RATE, label=name)
        return MISSING
