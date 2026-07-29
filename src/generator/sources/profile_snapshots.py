"""Источник `profile_snapshots` — помесячные снимки профиля клиента.

Снимков много и они разложены во времени: препроцессинг обязан выбрать
последний снимок с `profile_time_utc <= T` и никогда — будущий (§6).

Продукты появляются постепенно, поэтому список в снимке растёт. Это даёт
осмысленный `lifetime_product_count` на T (§24) и многозначное поле для §21.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator

from ..catalogs import EXTENDED_PRODUCTS
from ..clients import SyntheticClient
from ..config import GeneratorConfig
from ..edge_cases import Case, CaseLog, CasePlan, has
from ..records import RawRecord
from ..rng import derive_rng
from ..timing import iter_months, month_start

SOURCE = "profile_snapshots"

_NEGATIVE_AGE_VALUE = -1


def generate(
    clients: list[SyntheticClient],
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    for client in clients:
        yield from _for_client(client, config, plan, case_log)


def _for_client(
    client: SyntheticClient,
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    if has(client, Case.NO_PROFILE):
        case_log.record(
            Case.NO_PROFILE,
            source=SOURCE,
            client_id=client.client_id,
            record_id="-",
            note="ни одного снимка профиля: профиль на T собирается из MISSING",
        )
        return

    rng = derive_rng(config.seed, SOURCE, client.client_id)
    first_snapshot = client.active_from(config)
    if first_snapshot > config.history_end:
        return

    overflow = has(client, Case.MULTIVALUE_OVERFLOW)
    if overflow:
        case_log.record(
            Case.MULTIVALUE_OVERFLOW,
            source=SOURCE,
            client_id=client.client_id,
            record_id=client.cif,
            note=f"{len(EXTENDED_PRODUCTS)} продуктов — длиннее max_values_per_field",
        )

    acquired = _product_acquisition(client, config, rng, overflow=overflow)
    salary_index = 1.0 + rng.uniform(0.03, 0.14)  # годовая индексация зарплаты
    negative_age_month = month_start(plan.tie_break_moment.date())

    for snapshot_date in iter_months(first_snapshot, config.history_end):
        if snapshot_date < client.account_open:
            continue

        years_passed = (snapshot_date - client.account_open).days / 365.25
        salary = int(client.salary * (salary_index**years_passed))
        products = sorted(name for name, when in acquired.items() if when <= snapshot_date)

        age = client.age_at(snapshot_date)
        if has(client, Case.NEGATIVE_AGE) and snapshot_date == negative_age_month:
            age = _NEGATIVE_AGE_VALUE
            case_log.record(
                Case.NEGATIVE_AGE,
                source=SOURCE,
                client_id=client.client_id,
                record_id=f"{client.cif}@{snapshot_date.isoformat()}",
                note="возраст отрицательный — для этого поля значение невалидно (§17.3)",
            )

        payload = {
            "cif": client.cif,
            "snapshot_date": snapshot_date.isoformat(),
            "region": client.region,
            "employment": client.employment,
            "account_open_date": client.account_open.isoformat(),
            "salary": str(salary),
            "products": products,
            "age": age,
        }

        yield RawRecord(
            source=SOURCE,
            partition_date=snapshot_date,
            sort_key=client.cif,
            payload=payload,
        )


def _product_acquisition(
    client: SyntheticClient,
    config: GeneratorConfig,
    rng,
    *,
    overflow: bool,
) -> dict[str, date]:
    """Когда клиент получил каждый свой продукт. Первый — в день открытия счёта,
    остальные — случайно позже."""
    catalog = EXTENDED_PRODUCTS if overflow else list(client.products)
    acquired: dict[str, date] = {}
    span_days = max((config.history_end - client.account_open).days, 1)
    if overflow:
        # Переполнение должно быть видно задолго до T, поэтому все продукты
        # у такого клиента появляются в первой пятой части истории.
        span_days = max(span_days // 5, 1)
    for position, product in enumerate(catalog):
        if position == 0:
            acquired[product] = client.account_open
        else:
            acquired[product] = client.account_open + timedelta(days=rng.randint(1, span_days))
    return acquired
