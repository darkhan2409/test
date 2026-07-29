"""Граница T одинакова во всех местах, где она применяется — §14, §6 п.1.

Правило «не позже T» реализуют два разных прохода: индекс зон клиентов
(`ClientTimezoneIndex`, чтобы взять регион из последнего снимка до T) и
основной проход нормализации с отсечкой. Оба превращают дату снимка в
instant — и обязаны делать это одинаково.

Ошибка здесь тихая. Индекс, отсекающий по полуночи UTC, и проход,
локализующий ту же дату в полночь по Алматы, расходятся на пять часов:
снимок первого числа месяца попадает по разные стороны от T. Ничего не
падает, объём данных не меняется, часть клиентов просто получает зону
из умолчания вместо своего региона — и все их календарные признаки
уезжают на час. Заметить это можно только сверив два места между собой,
что и делает этот тест.

Класс ошибки — «два места реализуют одно правило по-разному». Он не
проявляется, пока правило не поменяют в одном из них.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.core.quarantine import Quarantine
from src.preprocessing.cutoff import CutoffFilter
from src.preprocessing.records import IdentifiedRecord
from src.preprocessing.schema import load_source_contracts
from src.preprocessing.timestamp_normalizer import (
    ClientTimezoneIndex,
    TimestampNormalizer,
    load_timestamp_policy,
)

UTC = timezone.utc
CUTOFF = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
RUN_AT = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)
CONFIG = Path("config")

# Полночь 1 февраля по Алматы (UTC+5) — это 31 января 19:00 UTC, то есть
# раньше T. Полночь 2 февраля — уже позже. Ровно на этой паре дат индекс
# и основной проход расходились.
BEFORE_T_LOCAL = "2026-02-01"
AFTER_T_LOCAL = "2026-02-02"
LONG_BEFORE_T = "2026-01-01"


@pytest.fixture(name="policy_and_registry")
def policy_and_registry_fixture():
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    return registry, load_timestamp_policy(CONFIG / "timestamp_policy.yaml", registry)


def snapshot(client_id: str, snapshot_date: str, region: str, line: int) -> IdentifiedRecord:
    return IdentifiedRecord(
        source="profile_snapshots",
        partition=f"profile_snapshots/{snapshot_date}.jsonl",
        line_number=line,
        source_record_id=f"CIF-{client_id}|{snapshot_date}",
        source_schema_version="1.0",
        client_ref=f"CIF-{client_id}",
        payload={
            "cif": f"CIF-{client_id}",
            "snapshot_date": snapshot_date,
            "region": region,
            "employment": "EMPLOYED",
            "account_open_date": "2020-01-01",
            "salary": "500000",
            "products": ["CARD"],
            "age": 35,
        },
        client_id=client_id,
    )


def run_pipeline(records, registry, policy, index):
    """Основной проход: нормализация времени + отсечка по T."""
    monitor = DataQualityMonitor()
    quarantine = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")
    normalizer = TimestampNormalizer(
        registry,
        policy,
        cutoff=CUTOFF,
        monitor=monitor,
        quarantine=quarantine,
        client_zones=index,
    )
    cutoff_filter = CutoffFilter(cutoff=CUTOFF, monitor=monitor)
    return list(cutoff_filter.apply(normalizer.normalize(records)))


def test_index_and_pipeline_agree_on_the_cutoff_boundary(policy_and_registry):
    """Снимок, переживший отсечку, обязан попасть в индекс зон — и наоборот.

    Проверяется именно совпадение двух множеств, а не конкретная дата:
    любое расхождение в трактовке границы разводит их.
    """
    registry, policy = policy_and_registry
    records = [
        snapshot("C000001", BEFORE_T_LOCAL, "ORAL", line=1),
        snapshot("C000002", AFTER_T_LOCAL, "ORAL", line=1),
        snapshot("C000003", LONG_BEFORE_T, "ATYRAU", line=1),
    ]

    index = ClientTimezoneIndex.build(
        records, registry=registry, policy=policy, cutoff=CUTOFF
    )
    survived = {record.client_id for record in run_pipeline(records, registry, policy, index)}

    assert set(index.zones) == survived
    assert survived == {"C000001", "C000003"}


def test_index_uses_the_last_snapshot_before_t_not_the_latest(policy_and_registry):
    """Регион берётся из последнего снимка **до** T, а не из самого свежего.

    Регионы у снимков разные специально: если индекс возьмёт снимок после T,
    зона окажется чужой, и подмену будет видно сразу, а не через сдвиг
    календарных признаков на час.
    """
    registry, policy = policy_and_registry
    records = [
        snapshot("C000001", LONG_BEFORE_T, "ALMATY", line=1),
        snapshot("C000001", BEFORE_T_LOCAL, "ATYRAU", line=1),
        snapshot("C000001", AFTER_T_LOCAL, "ORAL", line=1),
    ]

    index = ClientTimezoneIndex.build(
        records, registry=registry, policy=policy, cutoff=CUTOFF
    )

    assert index.zone_of("C000001") == "Asia/Atyrau"


def test_snapshot_date_becomes_local_midnight_not_utc_midnight(policy_and_registry):
    """Дата снимка превращается в полночь зоны источника.

    Это и есть спорные пять часов: `2026-02-01` в зоне Алматы — момент
    `2026-01-31T19:00Z`, который в окно наблюдения входит.
    """
    registry, policy = policy_and_registry
    records = [snapshot("C000001", BEFORE_T_LOCAL, "ORAL", line=1)]

    index = ClientTimezoneIndex.build(
        records, registry=registry, policy=policy, cutoff=CUTOFF
    )
    kept = run_pipeline(records, registry, policy, index)

    assert len(kept) == 1
    assert kept[0].timestamp_utc == datetime(2026, 1, 31, 19, 0, tzinfo=UTC)
    assert kept[0].timestamp_utc <= CUTOFF
