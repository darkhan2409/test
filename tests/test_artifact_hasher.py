"""Content hash состояния — §29.1, §30.

Хэш ценен ровно одним свойством: **изменилось состояние — изменился хэш**.
Всё остальное (что он SHA-256, что он в нижнем регистре) проверять незачем.
Опасен же обратный случай: раздел, который в пре-образ не попал. Тогда правка
конфига не меняет хэш, проверка при загрузке молчит, и данные посчитаны не тем
состоянием, что лежит рядом. Глазами это не видно никак — отсюда тест, который
трогает **каждый** раздел по очереди.

Второе проверяемое свойство — что несовпадение останавливает обработку, а не
превращается в предупреждение (§30).
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

import pytest

from src.preprocessing.artifact_hasher import (
    SPEC_30_SECTIONS,
    PreprocessingState,
    StateHashMismatchError,
    verify_state_hash,
)


def section(marker: str) -> dict[str, Any]:
    """Содержимое раздела. Разное у разных разделов — одинаковое не отличило бы
    «раздел попал в пре-образ» от «попал какой-то другой с тем же значением»."""
    return {"marker": marker, "nested": {"value": [marker, 1, True]}}


def state(**overrides: Any) -> PreprocessingState:
    """Состояние, где каждый раздел заполнен своим значением."""
    names = [item.name for item in dataclass_fields(PreprocessingState)]
    values = {name: section(name) for name in names}
    values.update(overrides)
    return PreprocessingState(**values)


# --------------------------------------------------------------------------- #
# Полнота состояния
# --------------------------------------------------------------------------- #


def test_spec_30_list_matches_the_regulation():
    """Перечень §30 не потерял пунктов.

    Список продублирован здесь намеренно — это единственное место, где он
    сверяется с регламентом, а не с самим собой. Без него пункт, вычеркнутый
    разом из `SPEC_30_SECTIONS` и из полей состояния, не уронил бы ни одной
    проверки: остальные тесты идут по тому, что в коде объявлено, и о
    вычеркнутом просто не узнают.

    Расходится с регламентом — правится код, а не список (`CLAUDE.md`).
    """
    from_regulation = {
        # «Source Contracts; mappings; Feature Schema» — маппингов у нас три,
        # каждый со своей версией в §30.
        "source_contracts",
        "identity_mapping",
        "event_mapping",
        "category_mapping",
        "feature_schema",
        "closed_set_domains",
        "bucket_field_domains",
        # «timestamp/calendar timezone policies»
        "timestamp_policy",
        "calendar_timezone_policy",
        # «dedup/sessionization configs»
        "dedup_config",
        "sessionization_config",
        "fx_config",
        "bucket_edges",
        "time_delta_edges",
        "numeric_rules",
        # «text/cutoff policies»
        "text_policy",
        "cutoff_policy",
        "serialization_config",
    }

    assert set(SPEC_30_SECTIONS) == from_regulation


def test_every_section_of_spec_30_is_hashed():
    """Каждый пункт §30 участвует в хэше — по одному изменению на раздел.

    Тест смотрит на `SPEC_30_SECTIONS`, а не на поля dataclass: сверяется
    перечень регламента с тем, что реально меняет хэш. Раздел, объявленный
    полем, но не попавший в `document()`, здесь и упадёт.
    """
    baseline = state().sha256()

    for name in SPEC_30_SECTIONS:
        changed = state(**{name: section("changed")}).sha256()
        assert changed != baseline, (
            f"раздел {name} не участвует в preprocessing_state_sha256 — "
            f"его изменение проходит мимо проверки §30"
        )


def test_sections_beyond_spec_30_are_hashed_too():
    """Разделы сверх §30 (hash_policy, run_policy и прочие) — тоже в хэше.

    Они добавлены не для полноты картины: `global_seed` меняет выборку, а
    `hash_policy` — все `event_id`. Незамеченное изменение здесь стоит столько
    же, сколько в разделе из перечня.
    """
    baseline = state().sha256()
    extra = [item.name for item in dataclass_fields(PreprocessingState)
             if item.name not in SPEC_30_SECTIONS]

    assert extra, "разделы сверх §30 исчезли — проверять нечего"
    for name in extra:
        assert state(**{name: section("changed")}).sha256() != baseline, (
            f"раздел {name} не участвует в preprocessing_state_sha256"
        )


def test_incomplete_state_cannot_be_built():
    """Состояние без раздела не собирается вовсе.

    Это и есть гарантия полноты: не проверка внутри хэшера, а невозможность
    позвать конструктор. Умолчание, добавленное полю позже, сняло бы её молча.
    """
    names = [item.name for item in dataclass_fields(PreprocessingState)]
    values = {name: section(name) for name in names[1:]}

    with pytest.raises(TypeError):
        PreprocessingState(**values)


# --------------------------------------------------------------------------- #
# Устойчивость хэша
# --------------------------------------------------------------------------- #


def test_hash_does_not_depend_on_key_order():
    """Порядок ключей внутри раздела на хэш не влияет (§29.1 п.2).

    Иначе один и тот же конфиг, разобранный в другом порядке — а порядок
    вставки в словарь именно таков, — дал бы разные хэши на разных прогонах.
    """
    direct = {"alpha": 1, "beta": {"x": 1, "y": 2}}
    shuffled = {"beta": {"y": 2, "x": 1}, "alpha": 1}
    assert list(direct) != list(shuffled), "словари совпали по порядку — проверять нечего"

    assert state(run_policy=direct).sha256() == state(run_policy=shuffled).sha256()


def test_hash_is_stable_between_runs():
    """Повторный расчёт того же состояния даёт тот же хэш."""
    assert state().sha256() == state().sha256()


# --------------------------------------------------------------------------- #
# Проверка при загрузке — §30
# --------------------------------------------------------------------------- #


def test_mismatch_blocks_processing():
    """Несовпадение — исключение, а не флаг и не предупреждение."""
    with pytest.raises(StateHashMismatchError, match="не совпал"):
        verify_state_hash(state(), "0" * 64)


def test_matching_hash_passes_silently():
    """Совпадение проходит молча: возвращать здесь нечего."""
    current = state()

    assert verify_state_hash(current, current.sha256()) is None
