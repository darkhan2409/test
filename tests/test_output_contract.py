"""Контракт выхода — §28 шаг 24, §2.2.

Это последняя точка, где ошибка цепочки ещё имеет имя: дальше значение
станет `token_id`, и «строковая сумма доехала до модели» превратится в
«модель почему-то плохо учится». Поэтому проверяется не то, что контракт
пропускает правильный выход (это видно в полном прогоне), а то, что он
**останавливает** каждый из перечисленных §2.2 случаев.

Отдельно проверяется отчёт: «контракт прошёл» на пустом потоке не значит
ничего, и число проверенных значений — единственный способ отличить
работающую проверку от отработавшей вхолостую.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.preprocessing.output_contract import OutputContractError, validate_output
from src.preprocessing.schema.constants import MISSING
from src.preprocessing.schema.feature_schema import (
    EventFeatureSchema,
    FeatureSchema,
    FieldSpec,
    FieldType,
    NumericSpec,
    NumericType,
    ProfileFeatureSchema,
    VocabularyPolicy,
)
from src.preprocessing.timeline_builder import TimelineRecord

UTC = timezone.utc
CUTOFF = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
MOMENT = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

BUCKET_DOMAIN = ("bucket_0", "bucket_1", MISSING)
CURRENCY_DOMAIN = ("KZT", "USD", MISSING)

# Domain бакет-поля в схеме не объявляется — его публикует BUILD (§11.1), и
# конфиг с ним модель отвергает. Поэтому здесь тот же путь, что в ENCODE:
# схема без domain плюс `resolve_bucket_domains`.
_DECLARED = FeatureSchema(
    version="1.0.0",
    events={
        "TRANSFER": EventFeatureSchema(
            event_type="TRANSFER",
            fields=(
                FieldSpec(
                    name="amount_base_bucket",
                    source_field="amount",
                    type=FieldType.BUCKET,
                    vocabulary_policy=VocabularyPolicy.BUCKET_CLOSED_SET,
                    priority=1,
                    numeric=NumericSpec(
                        numeric_type=NumericType.DECIMAL, signed=False, unit="KZT"
                    ),
                ),
                FieldSpec(
                    name="currency",
                    source_field="currency",
                    type=FieldType.CATEGORICAL,
                    vocabulary_policy=VocabularyPolicy.CLOSED_SET,
                    domain=CURRENCY_DOMAIN,
                    priority=2,
                ),
            ),
        )
    },
    profile=ProfileFeatureSchema(
        fields=(
            FieldSpec(
                name="region",
                source_field="region",
                type=FieldType.CATEGORICAL,
                vocabulary_policy=VocabularyPolicy.CLOSED_SET,
                domain=("ALMATY", MISSING),
                priority=1,
            ),
        )
    ),
)

SCHEMA = _DECLARED.resolve_bucket_domains({"amount_base_bucket": BUCKET_DOMAIN})
DOMAINS = {"amount_base_bucket": list(BUCKET_DOMAIN)}


def event(*, position: int = 0, fields: dict | None = None, **overrides) -> TimelineRecord:
    moment = overrides.pop("timestamp_utc", MOMENT + timedelta(minutes=position))
    record = TimelineRecord(
        source="core_payments",
        partition="core_payments/2026-01-01.jsonl",
        line_number=position,
        source_record_id=f"CP-{position:03d}",
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id="C000001",
        timestamp_utc=moment,
        calendar_timezone="Asia/Almaty",
        event_type="TRANSFER",
        event_id=f"{position:032d}",
        fields=fields if fields is not None else {"amount_base_bucket": "bucket_1"},
        schema_section="TRANSFER",
        ordering_key=f"{moment.isoformat()}|000010|CP-{position:03d}",
        position=position,
    )
    return replace(record, **overrides) if overrides else record


def validate(records):
    return validate_output(
        records, schema=SCHEMA, bucket_field_domains=DOMAINS, cutoff=CUTOFF
    )


# --------------------------------------------------------------------------- #
# Сторож на снятие гарантии
# --------------------------------------------------------------------------- #


def test_field_type_still_has_no_text_kind():
    """§23: свободный текст не проверяется, потому что он невыразим.

    Контракт §2.2 не ищет текст в значениях полей, и это верно ровно пока
    текстовое поле нельзя объявить. Гарантия — отсутствие текстового типа в
    `FieldType`, и она не вечная: §13 токенайзера прямо говорит, что
    BPE/subword может быть добавлена «в отдельной версии архитектуры». То
    есть текстовый тип это запланированное будущее, а не гипотеза.

    Здесь сторож ровно на это событие: как только тип появится, тест упадёт
    и напомнит, что §2.2 остался без проверки текста. Это не проверка
    невыразимого — это проверка самой гарантии, одна строка, срабатывающая
    один раз в жизни проекта.
    """
    assert set(FieldType) == {FieldType.CATEGORICAL, FieldType.BUCKET}, (
        "в FieldType появился новый тип поля: если он текстовый, §2.2 больше не "
        "защищён отсутствием возможности, и контракту нужна явная проверка текста "
        "(§23 препроцессинга, §13 токенайзера)"
    )


# --------------------------------------------------------------------------- #
# Отчёт: проверка отработала, а не промолчала
# --------------------------------------------------------------------------- #


def test_report_counts_what_was_actually_checked():
    """Пустой поток проходит любую проверку — отличить это можно только счётчиком."""
    report = validate([event(position=0), event(position=1)])

    assert report.events == 2
    assert report.bucket_values == 2
    assert validate([]).bucket_values == 0


# --------------------------------------------------------------------------- #
# Запреты §2.2
# --------------------------------------------------------------------------- #


def test_numeric_value_left_in_a_bucket_field_is_blocked():
    """Сумма, не размеченная §19, — первый пункт §2.2.

    Число в bucket-поле означает, что бакетизация до него не дошла; в
    словаре токенайзера оно стало бы отдельным значением-строкой.
    """
    with pytest.raises(OutputContractError, match="осталось числом"):
        validate([event(fields={"amount_base_bucket": Decimal("15000.50")})])


def test_bucket_value_outside_published_domain_is_blocked():
    """§33.4: значение вне опубликованного domain — critical."""
    with pytest.raises(OutputContractError, match="вне опубликованного domain"):
        validate([event(fields={"amount_base_bucket": "bucket_99"})])


def test_closed_set_value_outside_domain_is_blocked():
    """§16: ненормализованная валюта до токенайзера доходить не должна."""
    with pytest.raises(OutputContractError, match="вне closed-set domain"):
        validate([event(fields={"currency": "тенге"})])


def test_none_value_is_blocked():
    """`None` вместо `MISSING` — §15 не отработал."""
    with pytest.raises(OutputContractError, match="None"):
        validate([event(fields={"currency": None})])


def test_float_value_is_blocked():
    """float означает, что §17 не отработал: NaN пролез бы незаметно."""
    with pytest.raises(OutputContractError, match="float"):
        validate([event(fields={"currency": 1.5})])


def test_event_after_cutoff_is_blocked():
    """§14: событие позже T на выходе — утечка будущего."""
    with pytest.raises(OutputContractError, match="вне окна наблюдения"):
        validate([event(timestamp_utc=CUTOFF + timedelta(seconds=1))])


def test_duplicate_event_id_is_blocked():
    """Дубль, прошедший мимо §9, виден только по повтору `event_id`."""
    first = event(position=0)
    second = replace(event(position=1), event_id=first.event_id)

    with pytest.raises(OutputContractError, match="встречается дважды"):
        validate([first, second])


def test_field_outside_the_schema_is_blocked():
    """Поле не из Feature Schema — токенайзер о нём не знает (§11)."""
    with pytest.raises(OutputContractError, match="не объявлено в Feature Schema"):
        validate([event(fields={"merchant_name": "кофейня"})])


# --------------------------------------------------------------------------- #
# Порядок событий
# --------------------------------------------------------------------------- #


def test_event_without_ordering_key_is_blocked():
    """Без `ordering_key` порядок остался бы на усмотрение читателя (§2.2)."""
    with pytest.raises(OutputContractError, match="без position/ordering_key"):
        validate([replace(event(), ordering_key=None)])


def test_ordering_key_disagreeing_with_position_is_blocked():
    """Ключ уезжает в выход и объясняет порядок — разойдясь с ним, объясняет неверно.

    Данные подобраны так, что расхождение единственное: `position` растёт,
    а ключ убывает. Совпади они — проверка не отличила бы согласованный ключ
    от любого другого.
    """
    first, second = event(position=0), event(position=1)
    broken = replace(second, ordering_key="0000-00-00T00:00:00+00:00|000010|CP-001")
    assert broken.position > first.position, "position не растёт — проверять нечего"
    assert broken.ordering_key < first.ordering_key, "ключ не убывает — проверять нечего"

    with pytest.raises(OutputContractError, match="не возрастает"):
        validate([first, broken])
