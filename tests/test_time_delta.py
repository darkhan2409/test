"""Delta-канал — §25.2, §27 шаг 13.

Проверяется то, что решает границы и не видно в самих границах:

- дельта берётся **внутри клиента**; пара из событий разных клиентов не значит
  ничего, а расстояние между ними обычно велико и утащило бы верхние границы;
- у первого события клиента дельты нет — ноль вместо неё означал бы
  «предыдущее событие в тот же миг» и сдвинул бы первый бакет на всех сразу;
- `FIRST_EVENT` и `WINDOW_START` — вне обычных бакетов (§25.2): попасть в них
  расчётом нельзя.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.preprocessing.time_delta import (
    RESERVED,
    DeltaMethod,
    DeltaSampler,
    TimeDeltaConfig,
    TimeDeltaError,
    collect_deltas,
    fit_time_delta_edges,
)
from src.preprocessing.timeline_builder import TimelineRecord

UTC = timezone.utc
START = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)

CONFIG = TimeDeltaConfig(
    time_delta_edges_version="1.0.0",
    method=DeltaMethod.QUANTILE,
    bucket_count=4,
    sample_size=1000,
)


def event(*, client_id: str, offset: timedelta, position: int) -> TimelineRecord:
    moment = START + offset
    return TimelineRecord(
        source="core_payments",
        partition="core_payments/2026-01-01.jsonl",
        line_number=position,
        source_record_id=f"CP-{client_id}-{position:03d}",
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id=client_id,
        timestamp_utc=moment,
        calendar_timezone="Asia/Almaty",
        event_type="TRANSFER",
        event_id=f"{client_id}-{position:029d}",
        fields={},
        schema_section="TRANSFER",
        ordering_key=f"{moment.isoformat()}|000010|CP-{position:03d}",
        position=position,
    )


def deltas_of(records) -> list[Decimal]:
    sampler = DeltaSampler(sample_size=1000, global_seed=1)
    collect_deltas(records, sampler)
    return sorted(sampler.values())


# --------------------------------------------------------------------------- #
# Границы клиента
# --------------------------------------------------------------------------- #


def test_delta_is_never_taken_across_clients():
    """Пара из событий разных клиентов дельтой не является.

    Данные подобраны так, что ошибка была бы видна: расстояние между
    последним событием первого клиента и первым событием второго не совпадает
    ни с одной настоящей дельтой. Совпади оно — тест прошёл бы и на коде,
    который границу клиента не замечает.
    """
    records = [
        event(client_id="C000001", offset=timedelta(0), position=0),
        event(client_id="C000001", offset=timedelta(minutes=10), position=1),
        event(client_id="C000002", offset=timedelta(days=30), position=0),
        event(client_id="C000002", offset=timedelta(days=30, minutes=20), position=1),
    ]
    within = [Decimal(600), Decimal(1200)]
    across = Decimal(int(timedelta(days=30).total_seconds()) - 600)
    assert across not in within, "межклиентская дельта совпала с внутренней — тест пуст"

    assert deltas_of(records) == within


def test_first_event_of_a_client_has_no_delta():
    """Первому событию дельту взять неоткуда, и подставлять её нечем.

    Два клиента по два события дают ровно две дельты, а не четыре: у первых
    событий предыдущего нет, и нулём это не заменяется.
    """
    records = [
        event(client_id="C000001", offset=timedelta(0), position=0),
        event(client_id="C000001", offset=timedelta(minutes=5), position=1),
        event(client_id="C000002", offset=timedelta(days=1), position=0),
        event(client_id="C000002", offset=timedelta(days=1, minutes=5), position=1),
    ]

    values = deltas_of(records)

    assert values == [Decimal(300), Decimal(300)]
    assert Decimal(0) not in values


def test_unordered_timeline_is_blocking():
    """Отрицательная дельта — признак несортированного timeline (§13).

    Взять модуль значило бы посчитать границы по выдуманным данным: пара
    «позже → раньше» дала бы то же число, что и правильный порядок.
    """
    records = [
        event(client_id="C000001", offset=timedelta(minutes=10), position=0),
        event(client_id="C000001", offset=timedelta(0), position=1),
    ]

    with pytest.raises(TimeDeltaError, match="неупорядоченным"):
        deltas_of(records)


# --------------------------------------------------------------------------- #
# Артефакт §25.2
# --------------------------------------------------------------------------- #


def test_reserved_values_are_not_buckets():
    """`FIRST_EVENT` и `WINDOW_START` — вне обычных границ (§25.2).

    Они обязаны быть в domain (иначе токенайзер схлопнет их в `RARE` по
    `min_count`, §2.2) и обязаны не быть бакетами: расчёт границ их не
    порождает и породить не может.
    """
    records = [
        event(client_id="C000001", offset=timedelta(minutes=index * 7), position=index)
        for index in range(20)
    ]
    sampler = DeltaSampler(sample_size=1000, global_seed=1)
    collect_deltas(records, sampler)

    edges = fit_time_delta_edges(sampler, CONFIG)

    assert set(edges.labels()).isdisjoint(RESERVED)
    assert set(RESERVED) <= set(edges.domain())
    assert edges.domain()[: edges.bucket_count] == edges.labels()


def test_fit_without_a_single_delta_is_blocking():
    """Набор, где ни у кого нет второго события, границ не даёт.

    Молчаливый один бакет спрятал бы это: артефакт выглядел бы посчитанным.
    """
    sampler = DeltaSampler(sample_size=1000, global_seed=1)
    collect_deltas([event(client_id="C000001", offset=timedelta(0), position=0)], sampler)

    with pytest.raises(TimeDeltaError, match="нет ни одной дельты"):
        fit_time_delta_edges(sampler, CONFIG)


def test_edges_do_not_depend_on_the_order_of_offering():
    """Выборка и границы не зависят от порядка поступления (§29 п.5).

    Пары считаются по timeline, а выборка — priority sampling: перестановка
    клиентов местами меняет порядок предложений, но не множество дельт.
    """
    first = [
        event(client_id="C000001", offset=timedelta(minutes=index), position=index)
        for index in range(10)
    ]
    second = [
        event(client_id="C000002", offset=timedelta(hours=index), position=index)
        for index in range(10)
    ]

    direct = fit_time_delta_edges(_sampler(first + second), CONFIG)
    swapped = fit_time_delta_edges(_sampler(second + first), CONFIG)

    assert direct.state() == swapped.state()


def _sampler(records) -> DeltaSampler:
    sampler = DeltaSampler(sample_size=1000, global_seed=1)
    collect_deltas(records, sampler)
    return sampler
