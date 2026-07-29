"""Стабильные хэши — §29.1, пп. 9–13.

Та же причина, что и у `test_canonical.py`: `event_id` определяет
дедупликацию, tie-break и lineage. Если пре-образ изменится незаметно, весь
прошлый датасет станет несовместим с новым, а обнаружится это только по
расхождению golden-vectors.

Эталонные значения ниже заморожены первым прогоном. Их изменение — не повод
править тест, а сигнал, что изменилась hash policy: тогда обязаны смениться
`hash_policy_version` и `preprocessing_version` (§29.1 п.13).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.preprocessing.core.hashing import (
    HASH_POLICY,
    HashPolicyError,
    business_fingerprint,
    encode_timestamp,
    event_id,
    reservoir_seed,
    stable_digest,
)

UTC = timezone.utc

MOMENT = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)

FROZEN_TIMESTAMP = "1768469400000000"
FROZEN_EVENT_ID = "3237ced65b1febf36aa94c3eb5a9bcd8"
FROZEN_RESERVOIR_SEED = 13301011805813979997


def make_event_id(**overrides) -> str:
    fields = {
        "source_system": "core_payments",
        "source_record_id": "CP-000123-00007",
        "event_type": "TRANSFER",
        "event_timestamp": MOMENT,
    }
    fields.update(overrides)
    return event_id(**fields)


def test_length_prefix_prevents_concatenation_collision():
    """п.10: главное свойство length-prefix.

    С разделителем-символом ("ab" + "c") и ("a" + "bc") склеились бы в один
    пре-образ `abc` и дали одинаковый хэш — то есть два разных события
    получили бы один event_id.
    """
    assert stable_digest(("ab", "c")) != stable_digest(("a", "bc"))


def test_empty_field_is_not_the_same_as_missing_field():
    """п.10: пустое поле — это поле, а не его отсутствие."""
    assert stable_digest(("a",)) != stable_digest(("a", ""))


def test_event_id_is_128_bits_of_lowercase_hex():
    """п.11: ровно 32 hex-символа в нижнем регистре."""
    value = make_event_id()

    assert len(value) == 32
    assert value == value.lower()
    assert int(value, 16) >= 0  # строка действительно шестнадцатеричная


def test_event_id_is_stable_across_runs():
    """§8: одинаковая исходная запись обязана давать одинаковый id всегда.

    Замороженное значение ловит любое незаметное изменение пре-образа —
    порядка полей, кодировки, формата timestamp.
    """
    assert make_event_id() == FROZEN_EVENT_ID


@pytest.mark.parametrize(
    "override",
    [
        {"source_system": "card_processing"},
        {"source_record_id": "CP-000123-00008"},
        {"event_type": "PAYMENT"},
        {"event_timestamp": datetime(2026, 1, 15, 9, 30, 1, tzinfo=UTC)},
    ],
)
def test_every_preimage_field_affects_event_id(override):
    """§8: все четыре поля пре-образа реально участвуют в хэше.

    Если бы поле потерялось, две разные записи получили бы один id и одна
    исчезла бы при дедупликации.
    """
    assert make_event_id(**override) != FROZEN_EVENT_ID


def test_timestamp_encoded_as_epoch_microseconds():
    """п.10: время в пре-образе — целое число микросекунд, без float и локали."""
    assert encode_timestamp(MOMENT) == FROZEN_TIMESTAMP


def test_same_instant_in_another_timezone_gives_same_encoding():
    """п.10: хэш зависит от инстанта, а не от того, в какой зоне его записали."""
    from zoneinfo import ZoneInfo

    assert encode_timestamp(MOMENT.astimezone(ZoneInfo("Asia/Almaty"))) == FROZEN_TIMESTAMP


def test_naive_timestamp_rejected():
    """Наивное время здесь означает, что нормализация §12 не отработала:
    хэш посчитался бы по неизвестной зоне."""
    with pytest.raises(HashPolicyError) as error:
        encode_timestamp(datetime(2026, 1, 15, 9, 30))

    assert "без часового пояса" in str(error.value)


def test_non_string_part_rejected():
    """Молчаливое str(42) сделало бы хэш зависимым от типа входа."""
    with pytest.raises(HashPolicyError) as error:
        stable_digest(("core_payments", 42))

    assert "приведите его к строке явно" in str(error.value)


def test_business_fingerprint_depends_on_field_order():
    """§9.2: fingerprint versioned вместе с порядком полей.

    Перестановка полей обязана менять отпечаток — иначе смена конфигурации
    дедупликации прошла бы незамеченной.
    """
    fields = ("C000123", FROZEN_TIMESTAMP, "15000.50", "KZT", "OUT")

    assert business_fingerprint(fields) != business_fingerprint(tuple(reversed(fields)))


def test_reservoir_seed_is_uint64_and_stable():
    """п.12: seed укладывается в uint64 и не зависит от прогона."""
    seed = reservoir_seed("CP-000123-00007", 20260101)

    assert 0 <= seed < 2**64
    assert seed == FROZEN_RESERVOIR_SEED


@pytest.mark.parametrize(
    ("record_key", "global_seed"),
    [("CP-000123-00008", 20260101), ("CP-000123-00007", 20260102)],
)
def test_reservoir_seed_depends_on_both_inputs(record_key, global_seed):
    """п.12: и запись, и global_seed влияют на выборку.

    Если бы seed зависел только от записи, смена global_seed не меняла бы
    сэмпл, и повторный эксперимент был бы неотличим от исходного.
    """
    assert reservoir_seed(record_key, global_seed) != FROZEN_RESERVOIR_SEED


def test_global_seed_changes_the_high_bits():
    """Смена `global_seed` обязана менять **старшие** биты seed.

    Выборка (§19.6) упорядочивается по значению целиком, то есть решают
    старшие биты. `global_seed` сам по себе занимает пару десятков младших,
    и простой XOR оставлял бы старшие нетронутыми: два разных seed давали бы
    почти одинаковый сэмпл, а «повторим с другим seed» перестало бы быть
    независимым экспериментом. Поэтому seed перемешивается умножением.

    Проверяются соседние значения — именно на них слабость видна: у 20260101
    и 20260102 различается один младший бит.
    """
    keys = [f"CP-{index:06d}" for index in range(64)]
    left = [reservoir_seed(key, 20260101) >> 32 for key in keys]
    right = [reservoir_seed(key, 20260102) >> 32 for key in keys]

    assert all(a != b for a, b in zip(left, right))


def test_hash_policy_pins_the_format():
    """п.13: параметры формата зафиксированы в версионируемой политике."""
    assert HASH_POLICY["algorithm"] == "sha256"
    assert HASH_POLICY["preimage_framing"] == "length_prefix"
    assert HASH_POLICY["preimage_separator"] == "none"
    assert HASH_POLICY["event_id_field_order"] == [
        "source_system",
        "source_record_id",
        "event_type",
        "event_timestamp",
    ]
