"""Каноническая сериализация артефактов — §29.1, пп. 1–8.

Зачем это нужно именно так: `preprocessing_state_sha256` и сравнение
single/multi-worker имеют смысл только если одно и то же состояние даёт один
и тот же поток байт. Обычный `json.dumps` этого не гарантирует — порядок ключей
зависит от порядка вставки, `-0.0` печатается со знаком, а `NaN` вообще
превращается в невалидный JSON.

Формат зафиксирован в `SERIALIZATION_CONFIG`. Он сам входит в хэшируемое
состояние (§29.1 п.8): изменение формата обязано менять `preprocessing_version`,
иначе старые и новые артефакты будут молча несравнимы.

Реализация сознательно не пишет свой JSON-энкодер: дерево сначала проверяется
и нормализуется, а печатает его стандартный `json` с жёстко заданными
параметрами. Ручное экранирование строк — источник тонких багов, а стандартный
энкодер уже отлажен.
"""

from __future__ import annotations

import json
import math
from typing import Any

SERIALIZATION_CONFIG_VERSION = "1.0.0"

SERIALIZATION_CONFIG: dict[str, Any] = {
    "serialization_config_version": SERIALIZATION_CONFIG_VERSION,
    # §29.1 п.1
    "encoding": "utf-8",
    "bom": False,
    # §29.1 п.2 — сортировка по байтам UTF-8. Для UTF-8 порядок байт совпадает
    # с порядком кодовых точек, но сортируем явно по байтам, чтобы правило
    # читалось из кода, а не подразумевалось.
    "key_order": "utf8_byte_order",
    # §29.1 п.3
    "whitespace": "compact",
    "item_separator": ",",
    "key_separator": ":",
    # §29.1 п.4 — выбранный вариант из трёх допустимых
    "float_format": "shortest_round_trip_repr",
    # §29.1 п.5
    "negative_zero": "normalized_to_positive_zero",
    # §29.1 п.6
    "non_finite": "forbidden",
    # Не-ASCII печатаются как есть, а не как \uXXXX. Так надёжнее для
    # межреализационной идентичности: регистр hex в \uXXXX стандартом не
    # закреплён, а сырые UTF-8 байты однозначны.
    "non_ascii": "literal_utf8",
    "string_escapes": "json_standard_shortcuts",
    # Отказы сверх буквы §29.1. Регламент их не требует, но без них
    # детерминизм дырявый, поэтому они — часть версионируемого формата,
    # а не просто поведение кода.
    "non_string_keys": "forbidden",  # json склеил бы 1 и "1" в один ключ
    "unordered_collections": "forbidden",  # set/frozenset дают разный порядок
    "implicit_type_coercion": "forbidden",  # date/Decimal — преобразовывать явно
}


class CanonicalSerializationError(ValueError):
    """Значение нельзя сериализовать канонически.

    Это всегда блокирующая ошибка: молча «почти сериализовать» артефакт хуже,
    чем упасть, потому что расхождение вскроется только при сравнении хэшей.
    """


def canonical_bytes(value: Any) -> bytes:
    """Канонические байты значения — то, что идёт в SHA-256 и в файл."""
    return canonical_text(value).encode("utf-8")


def canonical_text(value: Any) -> str:
    """Канонический текст значения (без BOM, компактный, ключи отсортированы)."""
    normalized = _normalize(value, "$")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(SERIALIZATION_CONFIG["item_separator"], SERIALIZATION_CONFIG["key_separator"]),
        allow_nan=False,  # страховка: сюда уже не должно доходить не-конечных чисел
        sort_keys=False,  # порядок задан на этапе нормализации, по байтам UTF-8
        check_circular=True,
    )


def _normalize(value: Any, path: str) -> Any:
    """Проверить типы и привести значение к каноническому виду.

    Порядок проверок важен: `bool` — подкласс `int`, поэтому его разбираем
    раньше чисел.
    """
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value  # §29.1 п.7 — целые печатаются без плавающей точки

    if isinstance(value, float):
        return _normalize_float(value, path)

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return _normalize_dict(value, path)

    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]

    if isinstance(value, (set, frozenset)):
        raise CanonicalSerializationError(
            f"{path}: множество не имеет определённого порядка — приведите его "
            "к отсортированному списку до сериализации"
        )

    raise CanonicalSerializationError(
        f"{path}: тип {type(value).__name__} не сериализуется канонически; "
        "преобразуйте его явно (например, дату — в строку ISO-8601)"
    )


def _normalize_float(value: float, path: str) -> float:
    if math.isnan(value):
        raise CanonicalSerializationError(f"{path}: NaN запрещён в хэшируемом артефакте (§29.1 п.6)")
    if math.isinf(value):
        sign = "+Inf" if value > 0 else "-Inf"
        raise CanonicalSerializationError(
            f"{path}: {sign} запрещён в хэшируемом артефакте (§29.1 п.6)"
        )
    if value == 0.0:
        return 0.0  # §29.1 п.5 — снимает знак с -0.0
    return value


def _normalize_dict(value: dict[Any, Any], path: str) -> dict[str, Any]:
    for key in value:
        if not isinstance(key, str):
            # json молча превратил бы 1 и "1" в один и тот же ключ.
            raise CanonicalSerializationError(
                f"{path}: ключ {key!r} типа {type(key).__name__} — ключи объектов "
                "обязаны быть строками"
            )

    ordered = sorted(value, key=lambda key: key.encode("utf-8"))
    return {key: _normalize(value[key], f"{path}.{key}") for key in ordered}
