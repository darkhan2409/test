"""Источник `card_processing` — карточный процессинг.

Схема нарочно не похожа на `core_payments`:
- время в миллисекундах эпохи UTC (парсится совсем иначе, чем `15.01.2026 14:30`);
- сумма — целое в тиынах/центах, а не строка;
- валюта — ISO-4217 numeric (`398`, `840`);
- есть `merchant_id` — высококардинальное поле, которое §22 требует убрать
  или заменить категорией.

Блокировка карты (`BLK`) и снятие наличных (`WDR`) не имеют мерчанта, поэтому
соответствующие ключи просто отсутствуют: §15.1 — «поле неприменимо → поле
не добавляется», это не MISSING.

Краевые случаи на границе T живут здесь, а не в `core_payments`: там время
пишется с точностью до минуты, и «ровно T = 23:59:59» выразить нельзя.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterator

from ..catalogs import (
    CARD_OP_WEIGHTS,
    CURRENCY_ISO_NUMERIC,
    CURRENCY_WEIGHTS,
    FX_BASE_RATE,
    MCC_BY_CATEGORY,
    MERCHANT_CATEGORY_WEIGHTS,
    RARE_ONLY_CATEGORY,
    UNSEEN_ONLY_CATEGORY,
)
from ..clients import SyntheticClient
from ..config import GeneratorConfig
from ..edge_cases import Case, CaseLog, CasePlan, EventStream, has, planned_count
from ..records import RawRecord
from ..rng import derive_rng, weighted_choice
from ..timing import draw_local_datetime, epoch_millis, month_start, to_utc

SOURCE = "card_processing"
SCHEMA_VERSION = "1.2"

_TRANSACTIONS_PER_MONTH = 2.2

# Страна терминала коррелирует с валютой: покупка в USD обычно не в Казахстане.
_FOREIGN_COUNTRIES: dict[str, float] = {"TR": 0.35, "AE": 0.25, "RU": 0.2, "DE": 0.2}


def generate(
    clients: list[SyntheticClient],
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    for client in clients:
        yield from _background(client, config)
        yield from _injections(client, config, plan, case_log)


def _background(client: SyntheticClient, config: GeneratorConfig) -> Iterator[RawRecord]:
    rng = derive_rng(config.seed, SOURCE, client.client_id)
    active_from = client.active_from(config)
    if active_from > config.history_end:
        return

    months = max((config.history_end - active_from).days / 30.44, 0.5)
    natural = max(
        1, round(months * _TRANSACTIONS_PER_MONTH * client.activity * config.volume_scale)
    )
    count = planned_count(client, EventStream.CARD_PROCESSING, natural)

    for index in range(count):
        moment_local = draw_local_datetime(rng, client.timezone, active_from, config.history_end)
        operation = weighted_choice(rng, CARD_OP_WEIGHTS)

        payload: dict[str, Any] = {
            "rec_id": f"CRD-{client.cardholder_id}-{index:05d}",
            "cardholder_id": client.cardholder_id,
            "txn_time_ms": epoch_millis(moment_local),
            "op": operation,
            "schema_version": SCHEMA_VERSION,
        }

        if operation == "BLK":
            payload["block_reason"] = weighted_choice(
                rng, {"CLIENT_REQUEST": 0.6, "FRAUD_SUSPECTED": 0.25, "EXPIRED": 0.15}
            )
            payload["terminal_country"] = "KZ"
        else:
            currency = weighted_choice(rng, CURRENCY_WEIGHTS)
            amount_kzt = rng.lognormvariate(8.9, 1.0)
            amount = amount_kzt if currency == "KZT" else amount_kzt / FX_BASE_RATE[currency]

            payload["amount_minor"] = int(round(amount * 100))
            payload["currency_code"] = CURRENCY_ISO_NUMERIC[currency]
            payload["terminal_country"] = (
                "KZ" if currency == "KZT" else weighted_choice(rng, _FOREIGN_COUNTRIES)
            )

            if operation == "PUR":
                category = weighted_choice(rng, MERCHANT_CATEGORY_WEIGHTS)
                payload["merchant_category"] = category
                payload["mcc"] = MCC_BY_CATEGORY[category]
                payload["merchant_id"] = f"M-{rng.randrange(4000):04d}"
            else:  # WDR — снятие наличных: мерчанта нет, есть банкомат
                payload["atm_id"] = f"ATM-{rng.randrange(900):03d}"

        yield _record(payload, to_utc(moment_local))


def _injections(
    client: SyntheticClient,
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    if not client.cases or has(client, Case.NO_EVENTS):
        return

    rng = derive_rng(config.seed, SOURCE, "inject", client.client_id)

    if has(client, Case.TIE_BREAK_CROSS_SOURCE):
        yield _emit(
            client,
            case_log,
            Case.TIE_BREAK_CROSS_SOURCE,
            tag="tie-cross",
            moment_utc=plan.tie_break_cross_moment,
            category="GROCERY",
            amount_minor=777_00,
            note="парная запись к core_payments с тем же timestamp — порядок решает source_priority",
        )

    if has(client, Case.BOUNDARY_AT_T):
        yield _emit(
            client,
            case_log,
            Case.BOUNDARY_AT_T,
            tag="at-T",
            moment_utc=plan.boundary_at_t,
            category="FUEL",
            amount_minor=1234_00,
            note="timestamp == T, событие обязано войти в выборку",
        )

    if has(client, Case.BOUNDARY_AFTER_T):
        yield _emit(
            client,
            case_log,
            Case.BOUNDARY_AFTER_T,
            tag="after-T",
            moment_utc=plan.boundary_after_t,
            category="FUEL",
            amount_minor=4321_00,
            note="timestamp == T + 1 секунда, событие обязано быть отброшено",
        )

    if has(client, Case.RARE_VALUE):
        for index in range(plan.rare_value_count):
            yield _emit(
                client,
                case_log,
                Case.RARE_VALUE,
                tag=f"rare-{index}",
                moment_utc=plan.tie_break_moment + timedelta(days=index),
                category=RARE_ONLY_CATEGORY,
                amount_minor=rng.randrange(50_00, 900_00),
                note=f"категория встречается всего {plan.rare_value_count} раза до T → реже min_count",
            )

    if has(client, Case.UNSEEN_VALUE):
        for index in range(plan.unseen_value_count):
            moment = datetime(
                plan.unseen_from.year,
                plan.unseen_from.month,
                plan.unseen_from.day,
                12,
                index,
                tzinfo=plan.boundary_at_t.tzinfo,
            )
            yield _emit(
                client,
                case_log,
                Case.UNSEEN_VALUE,
                tag=f"unseen-{index}",
                moment_utc=moment,
                category=UNSEEN_ONLY_CATEGORY,
                amount_minor=rng.randrange(50_00, 900_00),
                note="категория появляется только после T → в TRAIN её нет",
            )


def _emit(
    client: SyntheticClient,
    case_log: CaseLog,
    case: Case,
    *,
    tag: str,
    moment_utc: datetime,
    category: str,
    amount_minor: int,
    note: str,
) -> RawRecord:
    record_id = f"CRDX-{client.cardholder_id}-{tag}"
    payload = {
        "rec_id": record_id,
        "cardholder_id": client.cardholder_id,
        "txn_time_ms": epoch_millis(moment_utc),
        "op": "PUR",
        "schema_version": SCHEMA_VERSION,
        "amount_minor": amount_minor,
        "currency_code": CURRENCY_ISO_NUMERIC["KZT"],
        "terminal_country": "KZ",
        "merchant_category": category,
        "mcc": MCC_BY_CATEGORY[category],
        "merchant_id": "M-0001",
    }
    case_log.record(
        case, source=SOURCE, client_id=client.client_id, record_id=record_id, note=note
    )
    return _record(payload, moment_utc)


def _record(payload: dict[str, Any], moment_utc: datetime) -> RawRecord:
    return RawRecord(
        source=SOURCE,
        partition_date=month_start(moment_utc.date()),
        sort_key=str(payload["rec_id"]),
        payload=payload,
    )
