"""Каноническая сериализация — §29.1, пп. 1–8.

Почему эти тесты существуют при установке «тестов по минимуму»: через
`canonical_bytes` проходит всё, что попадает в SHA-256 артефактов. Если поедет
порядок ключей или формат float, golden-vectors на этапе 4 скажут «не совпало»,
но не скажут где именно. Здесь поломка называется по имени.

Проверяется поведение, зафиксированное в `SERIALIZATION_CONFIG`, — то есть
контракт формата, а не детали реализации.
"""

from __future__ import annotations

import json

import pytest

from src.preprocessing.core.canonical import (
    SERIALIZATION_CONFIG,
    CanonicalSerializationError,
    canonical_bytes,
    canonical_text,
)


def test_utf8_without_bom_and_literal_cyrillic():
    """п.1: кодировка UTF-8 без BOM, не-ASCII пишутся как есть.

    Литеральный UTF-8 выбран вместо `\\uXXXX` намеренно: регистр hex в escape
    стандартом не закреплён, и другая реализация могла бы дать другие байты.
    """
    raw = canonical_bytes({"регион": "АЛМАТЫ"})

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw == '{"регион":"АЛМАТЫ"}'.encode("utf-8")


def test_keys_sorted_by_utf8_byte_order():
    """п.2: порядок ключей — байтовый, а не «алфавитный» на глаз.

    Заглавные латинские байты меньше строчных, кириллица идёт после всей
    латиницы. Порядок вставки на результат не влияет.
    """
    assert canonical_text({"я": 1, "b": 2, "a": 3, "Z": 4, "A": 5}) == (
        '{"A":5,"Z":4,"a":3,"b":2,"я":1}'
    )


def test_key_order_applies_at_every_level():
    """п.2: сортировка рекурсивная — вложенные объекты тоже упорядочены."""
    assert canonical_text({"outer": {"z": 1, "a": {"y": 1, "b": 2}}}) == (
        '{"outer":{"a":{"b":2,"y":1},"z":1}}'
    )


def test_insertion_order_does_not_change_bytes():
    """п.2: два словаря с одинаковым содержимым дают одинаковые байты.

    Это и есть свойство, ради которого сортировка нужна: в многопроцессной
    сборке словари собираются в разном порядке.
    """
    assert canonical_bytes({"a": 1, "b": 2, "c": 3}) == canonical_bytes({"c": 3, "b": 2, "a": 1})


def test_output_is_compact():
    """п.3: никаких незначащих пробелов между элементами."""
    assert canonical_text({"a": [1, 2], "b": 3}) == '{"a":[1,2],"b":3}'


def test_float_uses_shortest_round_trip_repr():
    """п.4: выбранный формат float — кратчайшее представление с round-trip.

    Проверяем оба свойства: и точное совпадение с `repr`, и то, что значения
    читаются обратно без потерь (иначе bucket_edges «поплывут» при перезагрузке).
    """
    edges = [0.1, 0.2, 0.30000000000000004, 1e16, 1.5e-8, 123456.789]
    serialized = canonical_text(edges)

    assert serialized == "[" + ",".join(repr(value) for value in edges) + "]"
    assert json.loads(serialized) == edges


def test_negative_zero_is_normalized():
    """п.5: -0.0 и 0.0 обязаны давать одинаковые байты.

    Иначе два математически одинаковых набора edges дали бы разный хэш.
    """
    assert canonical_text({"edge": -0.0}) == '{"edge":0.0}'
    assert canonical_bytes([-0.0, 0.0]) == canonical_bytes([0.0, 0.0])


@pytest.mark.parametrize(
    ("value", "expected_fragment"),
    [(float("nan"), "NaN"), (float("inf"), "+Inf"), (float("-inf"), "-Inf")],
)
def test_non_finite_floats_are_blocking_errors(value, expected_fragment):
    """п.6: NaN и бесконечности в хэшируемом артефакте — блокирующая ошибка.

    Молча пропустить их нельзя: стандартный json напечатал бы `NaN`, который
    невалиден по JSON, и другая реализация его просто не прочитает.
    """
    with pytest.raises(CanonicalSerializationError) as error:
        canonical_text({"bucket_edges": [1.0, value]})

    assert expected_fragment in str(error.value)


def test_integers_and_booleans_keep_their_type():
    """п.7: целые и булевы печатаются без плавающей точки, float — с ней."""
    assert canonical_text({"n": 42}) == '{"n":42}'
    assert canonical_text({"n": 2**70}) == '{"n":%d}' % 2**70
    assert canonical_text({"flag": True, "off": False}) == '{"flag":true,"off":false}'
    assert canonical_text({"x": 42.0}) == '{"x":42.0}'


def test_non_string_keys_rejected():
    """Сверх §29.1: json молча привёл бы 1 и "1" к одному ключу."""
    with pytest.raises(CanonicalSerializationError) as error:
        canonical_text({1: "a"})

    assert "ключи объектов" in str(error.value)


def test_unordered_collections_rejected():
    """Сверх §29.1: у множества нет определённого порядка обхода."""
    with pytest.raises(CanonicalSerializationError) as error:
        canonical_text({"products": {"CARD", "LOAN"}})

    assert "не имеет определённого порядка" in str(error.value)


def test_unsupported_types_rejected():
    """Сверх §29.1: дата обязана быть преобразована явно, а не «как-нибудь»."""
    from datetime import date

    with pytest.raises(CanonicalSerializationError) as error:
        canonical_text({"when": date(2026, 1, 1)})

    assert "не сериализуется" in str(error.value)


def test_error_message_points_at_the_broken_value():
    """Сообщение обязано называть путь: артефакты большие, искать вручную дорого."""
    with pytest.raises(CanonicalSerializationError) as error:
        canonical_text({"bucket_edges": {"amount": [1.0, float("nan")]}})

    assert "$.bucket_edges.amount[1]" in str(error.value)


def test_serialization_config_is_hashable_itself():
    """§29.1 п.8: описание формата само входит в хэшируемое состояние."""
    assert canonical_bytes(SERIALIZATION_CONFIG)
    assert SERIALIZATION_CONFIG["float_format"] == "shortest_round_trip_repr"
