"""Детерминированная случайность.

Каждый поток случайности выводится из global seed и стабильного ключа
(источник, клиент, номер записи), а не из общего последовательного состояния.
Тогда результат не зависит от порядка обхода клиентов и источников — это то же
требование, что §29 предъявляет к самому препроцессингу.
"""

from __future__ import annotations

import hashlib
import random


def derive_rng(seed: int, *parts: object) -> random.Random:
    """Вернуть независимый RNG для потока, заданного ключом `parts`."""
    digest = _digest(seed, parts)
    return random.Random(int.from_bytes(digest[:8], "big"))


def _digest(seed: int, parts: tuple[object, ...]) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(seed.to_bytes(8, "big", signed=True))
    for part in parts:
        raw = str(part).encode("utf-8")
        # length-prefix, а не символ-разделитель: «разделитель не встретится
        # в данных» недоказуемо (тот же принцип, что §29.1 п.10).
        hasher.update(len(raw).to_bytes(4, "big"))
        hasher.update(raw)
    return hasher.digest()


def weighted_choice(rng: random.Random, options: dict[str, float]) -> str:
    """Взвешенный выбор с фиксированным порядком ключей — иначе результат
    зависел бы от порядка обхода словаря."""
    keys = sorted(options)
    weights = [options[key] for key in keys]
    return rng.choices(keys, weights=weights, k=1)[0]
