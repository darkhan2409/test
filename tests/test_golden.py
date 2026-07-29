"""Golden-vector conformance set — §29.2.

Полный прогон эталона живёт в проверке 4.2: ему нужны данные, замороженные
артефакты и заморозка. Здесь — сверка, то есть то, что решает, засчитывать
прогон или нет.

Проверяется главное свойство: расхождение по **одному** артефакту ловится и
называется по имени. Общий хэш ответил бы «разошлось» без указания места, а
места разные по смыслу — данные, контракт с токенайзером, конфигурация.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from src.preprocessing.golden import (
    GOLDEN_EXPECTED_ITEMS,
    GoldenError,
    GoldenExpected,
    GoldenMismatchError,
    compare_expected,
    load_expected,
)

EVENT = {"client_id": "C000001", "event_id": "e" * 32, "fields": {"currency": "KZT"}}
PROFILE = {"client_id": "C000001", "profile_time_utc": "2026-01-31T23:59:59Z", "fields": {}}


def expected(**overrides) -> GoldenExpected:
    values = {
        "prepared_profile": [PROFILE],
        "prepared_events": [EVENT],
        "bucket_field_domains": {"amount_base_bucket": ["bucket_0", "MISSING"]},
        "preprocessing_state_sha256": "a" * 64,
    }
    values.update(overrides)
    return GoldenExpected(**values)


# --------------------------------------------------------------------------- #
# Состав §29.2 п.2
# --------------------------------------------------------------------------- #


def test_expected_list_matches_the_regulation():
    """Четыре артефакта §29.2 п.2 — дословно."""
    assert set(GOLDEN_EXPECTED_ITEMS) == {
        "prepared_profile",
        "prepared_events",
        "bucket_field_domains",
        "preprocessing_state_sha256",
    }


def test_expected_cannot_be_built_incomplete():
    """Артефакт нельзя не заполнить: эталон не должен замерзать неполным."""
    values = {item.name: "x" for item in dataclass_fields(GoldenExpected)[1:]}

    with pytest.raises(TypeError):
        GoldenExpected(**values)


# --------------------------------------------------------------------------- #
# Сверка
# --------------------------------------------------------------------------- #


def test_identical_run_passes():
    assert compare_expected(expected(), expected()) is None


def test_events_only_difference_is_caught_and_named():
    """Ключевой случай: изменились только события, хэш состояния тот же.

    Именно эту мутацию обязано ловить сравнение самих `prepared_events` —
    §30 её не заметит по построению, потому что состояние не изменилось.
    Проверено и на настоящем эталоне: сдвиг локального часа на единицу
    красит только `prepared_events`.
    """
    changed = expected(prepared_events=[{**EVENT, "fields": {"currency": "USD"}}])

    with pytest.raises(GoldenMismatchError) as error:
        compare_expected(changed, expected())

    text = str(error.value)
    assert "prepared_events" in text
    assert "prepared_profile" not in text
    assert "preprocessing_state_sha256" not in text


def test_message_points_at_the_record_and_the_field():
    """Расхождение указывает место, а не факт.

    На 374 событиях «не совпало» без указания записи и поля означает поиск
    вручную; с указанием — открыть и посмотреть.
    """
    changed = expected(prepared_events=[EVENT, {**EVENT, "event_id": "f" * 32}])

    with pytest.raises(GoldenMismatchError, match="записей 2 против 1"):
        compare_expected(changed, expected())

    changed = expected(prepared_events=[{**EVENT, "event_id": "f" * 32}])
    with pytest.raises(GoldenMismatchError, match="поле 'event_id'"):
        compare_expected(changed, expected())


def test_state_hash_difference_is_caught():
    """Хэш состояния сверяется и здесь, а не только при загрузке артефактов.

    §30 останавливает ENCODE, если состояние не совпало с **артефактами**;
    эталон ловит другое — что артефакты перезаморожены. Проверено сценарием:
    другое число бакетов delta-канала плюс перезаморозка красят ровно этот
    артефакт, потому что `time_delta_edges` входят в состояние, но в
    prepared-выход не попадают (§25.2 применяет их в токенайзере).
    """
    with pytest.raises(GoldenMismatchError, match="preprocessing_state_sha256"):
        compare_expected(expected(preprocessing_state_sha256="b" * 64), expected())


def test_all_differing_artifacts_are_reported_together():
    """Все разошедшиеся сразу: чинить придётся все."""
    changed = expected(
        prepared_profile=[{**PROFILE, "client_id": "C000002"}],
        preprocessing_state_sha256="b" * 64,
    )

    with pytest.raises(GoldenMismatchError) as error:
        compare_expected(changed, expected())

    text = str(error.value)
    assert "prepared_profile" in text and "preprocessing_state_sha256" in text


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #


def test_incomplete_frozen_set_is_refused(tmp_path: Path):
    """Эталон без одного из четырёх — не «частичный эталон», а неполный набор."""
    (tmp_path / "prepared_events.json").write_bytes(b"[]")

    with pytest.raises(GoldenError, match="набор §29.2 неполон"):
        load_expected(tmp_path)
