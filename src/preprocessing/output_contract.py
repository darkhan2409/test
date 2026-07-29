"""Финальная проверка выхода — §28 шаг 24, §2.2.

§2.2 перечисляет, чего на входе токенайзера остаться не должно. Список
читается как набор запретов, но по существу это последняя точка, где ошибка
цепочки ещё имеет имя. Дальше значение станет `token_id`, и «строковая сумма
доехала до модели» превратится в «модель почему-то плохо учится».

Проверяется здесь **не всё** из §2.2, и это осознанно. Часть запретов уже
невыразима: свободного текста нет, потому что в `FieldType` нет текстового
типа; `event_type` внутри `fields` отвергает модель §5. Проверять то, чего
нельзя объявить, значит писать код, который никогда не сработает, — и
однажды принять его срабатывание за ложную тревогу.

**Но у невыразимости есть срок.** §2.2 — граница с токенайзером, и защищает
она от изменений выше по течению, а §13 токенайзера прямо говорит, что
BPE/subword может быть добавлена «в отдельной версии архитектуры». То есть
текстовый тип поля — запланированное будущее, а не гипотеза. Поэтому
проверяется не текст в каждом значении, а **сама гарантия**: тест
`test_field_type_still_has_no_text_kind` падает ровно тогда, когда в
`FieldType` появляется новый тип, и напоминает, что этот раздел остался без
проверки текста. Сторож на конкретное событие, а не мёртвый код.

Проверяется то, что выразимо и потому возможно:

- значение bucket-поля — метка или `MISSING`, но не число и не строка суммы;
- значение closed-set поля — из опубликованного domain;
- `None`, `NaN` и `Inf` не доехали ни в одном поле;
- события не позже T;
- `event_id` уникален, то есть дубликат не проскочил мимо §9;
- порядок событий клиента задан и не зависит от читателя: `ordering_key`
  на месте и возрастает вместе с `position`;
- поля, помеченные `model_input: false`, до выхода не дошли.

Ошибка здесь всегда блокирующая и всегда означает баг цепочки, а не плохие
данные: плохие данные отправляет в карантин тот компонент, который их увидел
(§34). Сюда они дойти не могут.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from .feature_projection import ProjectedRecord
from .schema.constants import MISSING, PROFILE_SECTION
from .schema.feature_schema import FeatureSchema, FieldType, VocabularyPolicy

COMPONENT = "output_contract"


class OutputContractError(RuntimeError):
    """Выход не удовлетворяет контракту §2.2 — блокирующая ошибка."""


@dataclass
class OutputContractReport:
    """Что именно проверено. Нужен, чтобы «проверка прошла» не означало
    «проверка ничего не смотрела»: пустой поток тоже проходит любую проверку."""

    events: int = 0
    profiles: int = 0
    reference: int = 0
    """Записи без события и без профиля — справочники (курсы валют).

    Считаются отдельно, а не вместе с профилями: они материал §18, в
    `prepared_*` им места нет, и записанные в профили они завышали бы число
    клиентов втрое, не сломав ни одной проверки.
    """

    fields_checked: int = 0
    bucket_values: int = 0
    closed_set_values: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "profiles": self.profiles,
            "reference": self.reference,
            "fields_checked": self.fields_checked,
            "bucket_values": self.bucket_values,
            "closed_set_values": self.closed_set_values,
        }


def validate_output(
    records: Sequence[ProjectedRecord],
    *,
    schema: FeatureSchema,
    bucket_field_domains: Mapping[str, Sequence[str]],
    cutoff: datetime,
) -> OutputContractReport:
    """Проверить готовый выход против §2.2 и вернуть отчёт о проверенном."""
    report = OutputContractReport()
    specs = schema.field_specs()

    seen_events: set[str] = set()
    last_position: dict[str, int] = {}
    last_key: dict[str, str] = {}

    for record in records:
        if record.schema_section == PROFILE_SECTION:
            report.profiles += 1
            _check_fields(record, specs, bucket_field_domains, report)
            continue

        if record.event_type is None:
            report.reference += 1
            continue

        report.events += 1
        _check_event(record, cutoff, seen_events, last_position, last_key)
        _check_fields(record, specs, bucket_field_domains, report)

    return report


def _check_event(
    record: ProjectedRecord,
    cutoff: datetime,
    seen_events: set[str],
    last_position: dict[str, int],
    last_key: dict[str, str],
) -> None:
    if record.timestamp_utc is None or record.timestamp_utc > cutoff:
        raise OutputContractError(
            f"{record.raw_reference}: событие вне окна наблюдения (§2.2, §14)"
        )
    if not record.calendar_timezone:
        raise OutputContractError(
            f"{record.raw_reference}: событие без calendar_timezone (§2.2)"
        )

    event_id = record.event_id
    if not event_id:
        raise OutputContractError(f"{record.raw_reference}: событие без event_id (§8)")
    if event_id in seen_events:
        # §9 обязан был схлопнуть дубль. Здесь он означает пропуск, а не
        # плохие данные.
        raise OutputContractError(
            f"{record.raw_reference}: event_id {event_id} встречается дважды — "
            "дубликат прошёл мимо §9"
        )
    seen_events.add(event_id)

    client = record.client_id or ""
    position = getattr(record, "position", None)
    key = getattr(record, "ordering_key", None)
    if position is None or not key:
        raise OutputContractError(
            f"{record.raw_reference}: событие без position/ordering_key — порядок "
            "остался бы на усмотрение читателя (§13, §2.2)"
        )

    previous_position = last_position.get(client)
    if previous_position is not None:
        if position <= previous_position:
            raise OutputContractError(
                f"{record.raw_reference}: position {position} не возрастает после "
                f"{previous_position} — события клиента идут не в порядке §13"
            )
        # `ordering_key` и `position` обязаны говорить одно и то же: ключ
        # уезжает в выход и объясняет порядок, а расходившись с ним, объяснял
        # бы неверно.
        if key <= last_key[client]:
            raise OutputContractError(
                f"{record.raw_reference}: ordering_key {key!r} не возрастает после "
                f"{last_key[client]!r}, хотя position выросла — ключ и порядок разошлись"
            )
    last_position[client] = position
    last_key[client] = key


def _check_fields(
    record: ProjectedRecord,
    specs: Mapping[str, Any],
    bucket_field_domains: Mapping[str, Sequence[str]],
    report: OutputContractReport,
) -> None:
    for name, value in record.fields.items():
        spec = specs.get(name)
        if spec is None:
            raise OutputContractError(
                f"{record.raw_reference}: поле {name!r} не объявлено в Feature Schema "
                "(§11) — токенайзер о нём не знает"
            )
        if spec.vocabulary_policy is VocabularyPolicy.EXCLUDED:
            # §22: технический ID до модели не доходит. Дошёл — политика
            # исключения где-то не применилась.
            raise OutputContractError(
                f"{record.raw_reference}: поле {name!r} помечено excluded, но дошло "
                "до выхода (§22, §2.2)"
            )

        report.fields_checked += 1
        for item in _values_of(value):
            _check_value(record, name, item)
            if spec.type is FieldType.BUCKET:
                report.bucket_values += 1
                _check_bucket_value(record, name, item, bucket_field_domains)
            elif spec.vocabulary_policy is VocabularyPolicy.CLOSED_SET:
                report.closed_set_values += 1
                _check_closed_set_value(record, name, item, spec.domain or ())


def _values_of(value: Any) -> Iterable[Any]:
    """Многозначное поле проверяется поэлементно: один плохой элемент списка
    так же ломает контракт, как одиночное значение."""
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _check_value(record: ProjectedRecord, name: str, value: Any) -> None:
    if value is None:
        raise OutputContractError(f"{record.raw_reference}: {name} = None (§2.2, §15)")
    if isinstance(value, float):
        # §2.2 запрещает NaN на входе токенайзера, а §29.1 п.6 — в любом
        # хэшируемом артефакте. Дошедший float означает, что §17 не отработал.
        raise OutputContractError(
            f"{record.raw_reference}: {name} — float; после §17 значение обязано быть "
            "Decimal, а после §19 — меткой бакета"
        )


def _check_bucket_value(
    record: ProjectedRecord,
    name: str,
    value: Any,
    bucket_field_domains: Mapping[str, Sequence[str]],
) -> None:
    if isinstance(value, Decimal):
        raise OutputContractError(
            f"{record.raw_reference}: {name} осталось числом {value} — §19 не разметил "
            "его в бакет (§2.2 «значения, требующие numeric parsing, FX или clipping»)"
        )
    domain = bucket_field_domains.get(name)
    if domain is None:
        raise OutputContractError(
            f"{record.raw_reference}: для bucket-поля {name!r} не опубликован domain "
            "(§11.1, §2 п.5)"
        )
    if value not in domain:
        raise OutputContractError(
            f"{record.raw_reference}: {name} = {value!r} вне опубликованного domain "
            f"({len(domain)} значений) — §33.4 critical"
        )


def _check_closed_set_value(
    record: ProjectedRecord, name: str, value: Any, domain: Sequence[str]
) -> None:
    if value == MISSING:
        return
    if value not in domain:
        raise OutputContractError(
            f"{record.raw_reference}: {name} = {value!r} вне closed-set domain — "
            "§16 обязан был либо нормализовать значение, либо поставить MISSING"
        )
