"""Life-long признаки не зависят от порядка событий — §24.

Правило, а не везение. Расчёт §24 стоит **до** TimelineBuilder (§13, §26):
на этом шаге упорядоченного timeline ещё не существует, есть только множество
событий ≤ T. Признак, которому нужен порядок, получил бы здесь порядок
поступления записей — то есть порядок партиций, зависящий от нарезки потока
между воркерами. Результат менялся бы от числа процессов, что §29 запрещает.

Все шесть признаков §24 — счётчики и минимумы, обе операции коммутативны,
поэтому перенос расчёта раньше по цепечке безопасен. Этот тест закрепляет
условие, при котором перенос остаётся безопасным: **добавили признак, которому
нужен порядок — тест падает**. Чинить надо не тест, а место признака: ему
после TimelineBuilder.

Интерфейс аккумулятора устроен так, чтобы порядок было неоткуда взять — он
видит события по одному и последовательности не получает. Тест сторожит то,
что интерфейсом не выражается.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.event_mapper import load_event_mapping
from src.preprocessing.feature_projection import (
    PROFILE_SECTION,
    ProjectedRecord,
    load_feature_schema,
)
from src.preprocessing.profile_builder import (
    LIFELONG_FIELDS,
    ProfileBuilder,
    load_profile_policy,
)
from src.preprocessing.schema import load_source_contracts

UTC = timezone.utc
CONFIG = Path("config")
CUTOFF = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
START = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)


@pytest.fixture(name="setup")
def setup_fixture():
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    mapping = load_event_mapping(CONFIG / "event_mapping.yaml", registry)
    schema, _ = load_feature_schema(CONFIG / "feature_schema.yaml", registry, mapping.event_types)
    policy = load_profile_policy(
        CONFIG / "profile_policy.yaml", registry, schema, mapping.event_types
    )
    return schema, policy


def event(index: int, event_type: str, *, direction: str = "OUT") -> ProjectedRecord:
    return ProjectedRecord(
        source="core_payments",
        partition=f"core_payments/2024-{(index % 12) + 1:02d}-01.jsonl",
        line_number=index,
        source_record_id=f"CP-{index:05d}",
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id="C000001",
        timestamp_utc=START + timedelta(days=index),
        calendar_timezone="Asia/Almaty",
        event_type=event_type,
        event_id=f"{index:032x}",
        fields={"direction": direction},
        schema_section=event_type,
    )


def snapshot(index: int) -> ProjectedRecord:
    return ProjectedRecord(
        source="profile_snapshots",
        partition="profile_snapshots/2026-01-01.jsonl",
        line_number=index,
        source_record_id=f"CIF000001|2026-01-01",
        source_schema_version="1.0",
        client_ref="CIF000001",
        payload={
            "cif": "CIF000001",
            "snapshot_date": "2026-01-01",
            "account_open_date": "2020-06-15",
            "products": ["CARD", "DEPOSIT", "LOAN"],
        },
        client_id="C000001",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        calendar_timezone="Asia/Almaty",
        event_type=None,
        event_id=None,
        fields={"region": "ALMATY", "employment": "EMPLOYED"},
        schema_section=PROFILE_SECTION,
    )


def build(records, schema, policy) -> dict[str, object]:
    builder = ProfileBuilder(schema, policy, cutoff=CUTOFF, monitor=DataQualityMonitor())
    profiles = [r for r in builder.build(records) if r.schema_section == PROFILE_SECTION]
    assert len(profiles) == 1
    return {name: profiles[0].fields[name] for name in LIFELONG_FIELDS}


def sample_records() -> list[ProjectedRecord]:
    records: list[ProjectedRecord] = [snapshot(0)]
    records += [event(index, "CARD_PURCHASE") for index in range(1, 20)]
    records += [event(index, "TRANSFER", direction="IN") for index in range(20, 25)]
    records += [event(index, "APP_SESSION") for index in range(25, 40)]
    return records


def test_lifelong_features_are_invariant_under_permutation(setup):
    """Перестановка событий не меняет ни один из шести признаков §24.

    Проверяются несколько независимых перестановок: одна могла бы совпасть
    с исходной по случайности на маленьком наборе.
    """
    schema, policy = setup
    records = sample_records()
    expected = build(records, schema, policy)

    for seed in (1, 2, 3, 17):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        assert build(shuffled, schema, policy) == expected, f"перестановка seed={seed}"


def test_reversed_order_gives_the_same_result(setup):
    """Обратный порядок — крайний случай перестановки.

    Именно он ловит признак «первое/последнее по порядку поступления»:
    случайная перестановка могла бы оставить первый элемент на месте.
    """
    schema, policy = setup
    records = sample_records()

    assert build(list(reversed(records)), schema, policy) == build(records, schema, policy)


def test_earliest_event_defines_first_seen_age_not_the_first_encountered(setup):
    """`first_seen_age` — минимум по времени, а не первое встреченное событие.

    Самое раннее событие подаётся последним: если признак берёт «первое, что
    пришло», значение окажется меньше, и подмену будет видно числом.
    """
    schema, policy = setup
    late = event(10, "CARD_PURCHASE")
    early = event(1, "CARD_PURCHASE")

    result = build([snapshot(0), late, early], schema, policy)

    assert result["first_seen_age_bucket"] == Decimal((CUTOFF - early.timestamp_utc).days)


def test_first_topup_uses_the_earliest_incoming_event(setup):
    """`first_topup_age` считается по самому раннему **входящему** движению.

    Исходящие события того же типа стоят раньше по времени: если предикат
    направления потеряется, значение уедет к ним.
    """
    schema, policy = setup
    outgoing = event(1, "TRANSFER", direction="OUT")
    incoming = event(5, "TRANSFER", direction="IN")

    result = build([snapshot(0), outgoing, incoming], schema, policy)

    assert result["first_topup_age_bucket"] == Decimal((CUTOFF - incoming.timestamp_utc).days)


def dated_snapshot(day: str, region: str, index: int) -> ProjectedRecord:
    record = snapshot(index)
    return type(record)(**{
        **record.__dict__,
        "source_record_id": f"CIF000001|{day}",
        "timestamp_utc": datetime.fromisoformat(day).replace(tzinfo=UTC),
        "payload": {**record.payload, "snapshot_date": day},
        "fields": {**record.fields, "region": region},
    })


def build_profile(records, schema, policy) -> ProjectedRecord:
    builder = ProfileBuilder(schema, policy, cutoff=CUTOFF, monitor=DataQualityMonitor())
    profiles = [r for r in builder.build(records) if r.schema_section == PROFILE_SECTION]
    assert len(profiles) == 1
    return profiles[0]


def test_latest_snapshot_wins_not_the_first_encountered(setup):
    """§6 п.1: берётся последний снимок до T, а не первый попавшийся.

    Снимки подаются в порядке, обратном хронологии: если выбор делается по
    порядку поступления, победит январский 2024-го, и профиль клиента
    окажется двухлетней давности. Регионы у снимков разные — подмену видно
    значением, а не только идентификатором.
    """
    schema, policy = setup
    records = [
        dated_snapshot("2026-01-01", "ASTANA", 3),
        dated_snapshot("2025-01-01", "SHYMKENT", 2),
        dated_snapshot("2024-01-01", "ALMATY", 1),
    ]

    profile = build_profile(records, schema, policy)

    assert profile.source_record_id == "CIF000001|2026-01-01"
    assert profile.fields["region"] == "ASTANA"


def test_snapshot_choice_is_invariant_under_permutation(setup):
    """Порядок поступления снимков на выбор не влияет."""
    schema, policy = setup
    records = [
        dated_snapshot("2024-01-01", "ALMATY", 1),
        dated_snapshot("2026-01-01", "ASTANA", 3),
        dated_snapshot("2025-01-01", "SHYMKENT", 2),
    ]

    chosen = {
        seed: build_profile(
            random.Random(seed).sample(records, len(records)), schema, policy
        ).source_record_id
        for seed in (1, 2, 3, 4, 5)
    }

    assert set(chosen.values()) == {"CIF000001|2026-01-01"}


def test_snapshots_with_equal_timestamps_are_resolved_deterministically(setup):
    """Два снимка с одинаковым моментом решаются `source_record_id`.

    После дедупликации (§9) такой пары быть не должно: ключ снимка — это
    `(cif, snapshot_date)`. Но `max()` без второго ключа вернул бы первый
    максимальный **в порядке поступления**, то есть в порядке партиций,
    зависящем от нарезки потока. Проверка стоит именно поэтому: цена
    защиты — один ключ сортировки, цена её отсутствия — недетерминизм,
    который проявится только когда предусловие сломается.
    """
    schema, policy = setup
    first = dated_snapshot("2026-01-01", "ASTANA", 1)
    twin = type(first)(**{
        **first.__dict__,
        "source_record_id": "CIF000001|2026-01-01-b",
        "fields": {**first.fields, "region": "ORAL"},
    })

    forward = build_profile([first, twin], schema, policy)
    backward = build_profile([twin, first], schema, policy)

    assert forward.source_record_id == backward.source_record_id
    assert forward.fields["region"] == backward.fields["region"]


def test_counts_do_not_depend_on_order(setup):
    """Счётчики §24 считают множество, а не последовательность."""
    schema, policy = setup
    records = sample_records()

    result = build(records, schema, policy)

    assert result["lifetime_event_count_bucket"] == Decimal(39)
    # Транзакции — только денежные типы: сессии в счётчик не входят.
    assert result["lifetime_transaction_count_bucket"] == Decimal(24)
    assert result["lifetime_product_count_bucket"] == Decimal(3)
