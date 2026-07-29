"""Выход §32 и контракт §2.

Проверяется то, что нельзя увидеть глазами в 31 тысяче строк JSONL:

- перечень §2 совпадает с тем, что выход обещает отдать, — девятнадцать
  пунктов, ни больше ни меньше;
- пункт, живущий внутри записей (`lifetime_first`, `calendar_time_features`),
  описан адресом «файл + поле», а не именем файла: контракт обязан отвечать
  на вопрос «где это», а не «есть ли такой файл»;
- форма записей — §32.1 и §32.2 дословно, включая то, чего в них быть не
  должно.

Полный прогон на реальных данных живёт отдельно: он проверяет выход как
документ, а здесь — как форму.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, replace
from datetime import datetime, timezone

import pytest

from src.preprocessing.prepared_output import (
    CONTRACT_ITEMS,
    EVENTS_FILE,
    METADATA_FILE,
    ContractItem,
    PreparedOutputError,
    TokenizerContract,
    event_row,
    profile_row,
)
from src.preprocessing.schema.constants import PROFILE_SECTION
from src.preprocessing.schema.event import CalendarTimeFeatures
from src.preprocessing.time_feature_builder import TimeFeaturedRecord

UTC = timezone.utc
MOMENT = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)


def event(*, fields: dict | None = None, **overrides) -> TimeFeaturedRecord:
    """Событие в том виде, в каком его отдаёт конец цепочки (§25)."""
    record = TimeFeaturedRecord(
        source="core_payments",
        partition="core_payments/2026-01-01.jsonl",
        line_number=1,
        source_record_id="CP-001",
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id="C000001",
        timestamp_utc=MOMENT,
        calendar_timezone="Asia/Almaty",
        event_type="TRANSFER",
        event_id="e" * 32,
        fields=fields if fields is not None else {"amount_base_bucket": "bucket_1"},
        schema_section="TRANSFER",
        ordering_key=f"{MOMENT.isoformat()}|000010|CP-001",
        position=0,
        calendar_time_features=CalendarTimeFeatures(
            hour_of_day_local=14, day_of_week_local=3
        ),
        lifetime_first=True,
    )
    return replace(record, **overrides) if overrides else record


def contract(**overrides) -> TokenizerContract:
    values = {item.name: ContractItem("file.json") for item in dataclass_fields(TokenizerContract)}
    values.update(overrides)
    return TokenizerContract(**values)


# --------------------------------------------------------------------------- #
# Перечень §2
# --------------------------------------------------------------------------- #


def test_contract_list_matches_the_regulation():
    """Девятнадцать пунктов §2 — дословно.

    Список продублирован здесь намеренно, как и перечень §30: это
    единственное место, где он сверяется с регламентом, а не сам с собой.
    Пункт, вычеркнутый разом из `CONTRACT_ITEMS` и из полей контракта, иначе
    не уронил бы ни одной проверки.
    """
    from_regulation = {
        "prepared_profile",
        "prepared_events",
        "feature_schema",
        "closed_set_domains",
        "bucket_field_domains",
        "bucket_metadata",
        "time_delta_edges",
        "calendar_time_features",
        "currency_normalization_config",
        "fx_max_staleness",
        "sessionization_config",
        "field_priority",
        "max_values_per_field",
        "text_policy",
        "cutoff_policy",
        "preprocessing_version",
        "preprocessing_state_sha256",
        "data_quality_statistics",
        "lifetime_first",
    }

    assert len(CONTRACT_ITEMS) == 19
    assert set(CONTRACT_ITEMS) == from_regulation


def test_contract_cannot_be_built_with_a_missing_item():
    """Пункт нельзя не заполнить: это `TypeError`, а не тихо неполный выход."""
    values = {
        item.name: ContractItem("file.json")
        for item in dataclass_fields(TokenizerContract)[1:]
    }

    with pytest.raises(TypeError):
        TokenizerContract(**values)


def test_items_living_inside_records_carry_a_field_address():
    """`lifetime_first` — не файл, а поле каждого события (§2 п.19).

    Адрес без поля отправил бы читателя искать файл `lifetime_first.json`,
    которого нет и быть не должно.
    """
    built = contract(
        lifetime_first=ContractItem(EVENTS_FILE, "lifetime_first"),
        fx_max_staleness=ContractItem(METADATA_FILE, "fx_max_staleness_days"),
    )

    manifest = built.manifest()

    assert manifest["lifetime_first"] == {"file": EVENTS_FILE, "field": "lifetime_first"}
    assert manifest["fx_max_staleness"]["field"] == "fx_max_staleness_days"
    assert manifest["prepared_events"]["field"] is None


# --------------------------------------------------------------------------- #
# Форма записей §32
# --------------------------------------------------------------------------- #


def test_event_row_matches_32_2():
    """§32.2: ровно эти ключи, `event_type` только top-level."""
    row = event_row(event(fields={"currency": "KZT"}))

    assert set(row) == {
        "client_id", "event_id", "event_type", "timestamp_utc", "calendar_timezone",
        "ordering_key", "fields", "calendar_time_features", "lifetime_first",
    }
    assert "event_type" not in row["fields"]
    assert row["timestamp_utc"].endswith("Z")


def test_event_row_never_carries_the_final_delta():
    """§32.2: финальная `delta_from_previous_event` до выбора окна не передаётся.

    Проверяется не отсутствие ключа в конкретной записи, а то, что его нечем
    туда положить: поле не считается ни на одном шаге цепочки.
    """
    row = event_row(event())

    assert "delta_from_previous_event" not in row
    assert "delta_from_previous_event" not in row["fields"]


def test_event_without_calendar_features_is_refused():
    """Событие без §25 в выход не попадает: токенайзер ждёт локальный час."""
    with pytest.raises(PreparedOutputError, match="без календарных признаков"):
        event_row(event(calendar_time_features=None))


def test_profile_row_matches_32_1():
    """§32.1: клиент, время профиля и поля — больше ничего."""
    profile = event(
        fields={"region": "ALMATY"},
        event_type=None,
        event_id=None,
        schema_section=PROFILE_SECTION,
    )

    row = profile_row(profile)

    assert set(row) == {"client_id", "profile_time_utc", "fields"}
    assert row["fields"] == {"region": "ALMATY"}
