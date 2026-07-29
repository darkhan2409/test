"""Стабильные хэши — §29.1, пп. 9–13.

Один именованный алгоритм на весь pipeline: SHA-256. Встроенный `hash()`
запрещён — он рандомизирован между процессами, и `event_id` менялся бы от
запуска к запуску.

Ключевая тонкость — как склеивать поля в пре-образ. Разделитель-символ не
годится: утверждение «этот символ не встретится в данных» недоказуемо, а если
встретится, два разных набора полей дадут один хэш. Поэтому перед каждым полем
пишется его длина в байтах (§29.1 п.10):

    ("ab", "c")  →  [8 байт: 2]ab[8 байт: 1]c
    ("a", "bc")  →  [8 байт: 1]a[8 байт: 2]bc

Одинаковой склейки `abc` больше не получается — хэши разные.

`HASH_POLICY` описывает все параметры формата и входит в
`preprocessing_state_sha256` (§29.1 п.13): изменение любого из них обязано
менять `preprocessing_version`, иначе старые `event_id` перестанут совпадать
с новыми молча.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

UTC = timezone.utc

HASH_POLICY_VERSION = "1.1.0"

_LENGTH_PREFIX_BYTES = 8
_LENGTH_PREFIX_ENDIANNESS = "big"
_EVENT_ID_BYTES = 16  # 128 бит = 32 hex-символа (§29.1 п.11)
_RESERVOIR_SEED_BYTES = 8
_UINT64_MASK = (1 << 64) - 1

# Нечётная константа золотого сечения (2**64 / φ). Ею домножается global_seed
# перед XOR: сам по себе seed занимает два-три десятка младших бит, и простой
# XOR менял бы только их. Выборка же упорядочивается по значению целиком, то
# есть по старшим битам — и при простом XOR два разных global_seed давали бы
# почти одинаковую выборку. Умножение на нечётную константу — обратимая
# операция, размазывающая seed по всем 64 битам.
_SEED_MIXER = 0x9E3779B97F4A7C15

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)

# Порядок полей пре-образа event_id зафиксирован §8 и менять его нельзя
# отдельно от версии политики.
EVENT_ID_FIELD_ORDER: tuple[str, ...] = (
    "source_system",
    "source_record_id",
    "event_type",
    "event_timestamp",
)

HASH_POLICY: dict[str, Any] = {
    "hash_policy_version": HASH_POLICY_VERSION,
    # §29.1 п.9
    "algorithm": "sha256",
    "builtin_hash": "forbidden",
    # §29.1 п.10
    "preimage_framing": "length_prefix",
    "preimage_separator": "none",
    "length_prefix_bytes": _LENGTH_PREFIX_BYTES,
    "length_prefix_endianness": _LENGTH_PREFIX_ENDIANNESS,
    "length_counts": "utf8_bytes",
    "part_encoding": "utf-8",
    "timestamp_encoding": "epoch_microseconds_decimal",
    # Domain separation (метка назначения первым полем пре-образа) рассмотрена
    # и сознательно опущена — это решение, а не недосмотр. Причины: length-prefix
    # даёт однозначную разборку пре-образа, поэтому коллизия между event_id,
    # fingerprint и reservoir key невозможна без совпадения всей
    # последовательности полей; а §29.1 п.11 фиксирует состав пре-образа
    # event_id ровно по §8, и лишнее поле было бы отступлением от регламента.
    "domain_separation": "omitted_by_decision",
    "domain_separation_reason": (
        "length-prefix framing is unambiguous; §29.1 п.11 fixes the event_id "
        "preimage to the §8 field list only"
    ),
    # §29.1 п.11
    "event_id_bits": _EVENT_ID_BYTES * 8,
    "event_id_format": "lowercase_hex",
    "event_id_field_order": list(EVENT_ID_FIELD_ORDER),
    # §29.1 п.12
    "reservoir_seed_bytes": _RESERVOIR_SEED_BYTES,
    "reservoir_seed_endianness": "big",
    "reservoir_seed_mixer": f"0x{_SEED_MIXER:016X}",
    "reservoir_seed_formula": (
        "uint64_be(sha256(record_key)[:8]) xor ((global_seed * mixer) mod 2**64)"
    ),
    "reservoir_seed_mixer_reason": (
        "global_seed occupies only the low bits; ordering is by the full 64-bit value, "
        "so a plain XOR would leave different seeds producing nearly identical samples"
    ),
}


class HashPolicyError(ValueError):
    """Нарушение hash policy: некорректный вход для стабильного хэша."""


def stable_digest(parts: Sequence[str]) -> bytes:
    """SHA-256 по length-prefix пре-образу из полей `parts`.

    Поля обязаны быть строками: молчаливое `str(42)` сделало бы хэш зависимым
    от того, пришло число как `int` или как строка.
    """
    hasher = hashlib.sha256()
    for index, part in enumerate(parts):
        if not isinstance(part, str):
            raise HashPolicyError(
                f"поле {index} имеет тип {type(part).__name__}; приведите его к строке явно "
                "(timestamp — через encode_timestamp)"
            )
        raw = part.encode("utf-8")
        hasher.update(len(raw).to_bytes(_LENGTH_PREFIX_BYTES, _LENGTH_PREFIX_ENDIANNESS))
        hasher.update(raw)
    return hasher.digest()


def stable_hex(parts: Sequence[str]) -> str:
    """Полный SHA-256 в нижнем регистре — для fingerprint и прочих ключей."""
    return stable_digest(parts).hex()


def content_hex(payload: bytes) -> str:
    """SHA-256 готового потока байт — content hash артефакта (§30).

    Length-prefix (§29.1 п.10) здесь не нужен и был бы неверен: он решает
    задачу однозначной склейки нескольких полей, а тут пре-образ — уже готовый
    канонический документ (§29.1 пп. 1–8), один и неделимый.

    Функция живёт рядом со `stable_digest` не для удобства: §29.1 п.9 требует
    **один** именованный алгоритм на весь pipeline, и держать выбор алгоритма
    в двух модулях значит однажды поменять его в одном.
    """
    return hashlib.sha256(payload).hexdigest()


def encode_timestamp(moment: datetime) -> str:
    """Timestamp для пре-образа: целое число микросекунд эпохи (§29.1 п.10).

    Наивный datetime отвергается: на этом уровне он означает, что нормализация
    времени (§12) не отработала, и хэш посчитался бы по неизвестной зоне.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise HashPolicyError(
            "timestamp без часового пояса: приведите его к aware UTC до хэширования (§12)"
        )
    return str((moment.astimezone(UTC) - _EPOCH) // _MICROSECOND)


def event_id(
    *,
    source_system: str,
    source_record_id: str,
    event_type: str,
    event_timestamp: datetime,
) -> str:
    """Стабильный `event_id` — §8 и §29.1 п.11.

    Одинаковая исходная запись при повторном запуске обязана дать тот же id:
    ни времени запуска, ни номера воркера, ни случайности здесь нет.
    Возвращает 32 hex-символа (первые 128 бит SHA-256).
    """
    digest = stable_digest(
        (source_system, source_record_id, event_type, encode_timestamp(event_timestamp))
    )
    return digest[:_EVENT_ID_BYTES].hex()


def business_fingerprint(parts: Sequence[str]) -> str:
    """Versioned fingerprint бизнес-дубликата — §9.2.

    Набор и порядок полей задаёт `dedup_config` конкретного источника; здесь
    только фиксируется способ склейки, общий для всего pipeline.
    """
    return stable_hex(parts)


def reservoir_seed(record_key: str, global_seed: int) -> int:
    """Seed для детерминированного reservoir sampling — §19.6 и §29.1 п.12.

    Зависит только от содержимого записи и глобального seed, поэтому выборка
    не меняется от числа воркеров и порядка завершения задач.

    Структура задана §29.1 п.12: первые 8 байт SHA-256 record key как uint64
    big-endian, скомбинированные с `global_seed` утверждённой формулой.
    Формула — XOR с перемешанным seed (см. `_SEED_MIXER`).
    """
    digest = stable_digest((record_key,))
    record_component = int.from_bytes(digest[:_RESERVOIR_SEED_BYTES], "big")
    seed_component = (global_seed * _SEED_MIXER) & _UINT64_MASK
    return (record_component ^ seed_component) & _UINT64_MASK
