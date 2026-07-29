"""Расчёт границ по выборке — общая арифметика §19.3 и §25.2.

Границы считают два **разных** компонента: бакетизация числовых полей события
(§19) и бакетизация временных дельт (§25.2). Регламент требует держать их
раздельно — у них разные артефакты, разные версии и разный момент применения
(дельту трансформирует токенайзер после truncation). Но сама арифметика у них
обязана быть одной: значение, стоящее ровно на границе, должно попадать в один
и тот же бакет в обоих случаях, иначе две реализации одного правила однажды
разойдутся на единицу.

Поэтому здесь только счёт: ни артефактов, ни версий, ни имён бакетов.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence


def quantile_edges(values: Sequence[Decimal], count: int) -> list[Decimal]:
    """Внутренние границы по квантилям отсортированной выборки.

    Берётся значение выборки в позиции квантиля, а не интерполяция между
    соседями: интерполяция породила бы границу, которой в данных нет, и на
    целочисленном поле дала бы дробную границу.
    """
    size = len(values)
    edges: list[Decimal] = []
    for index in range(1, count):
        position = (size * index) // count
        edges.append(values[min(position, size - 1)])
    return edges


def equal_width_edges(values: Sequence[Decimal], count: int) -> list[Decimal]:
    """Внутренние границы равной ширины между минимумом и максимумом выборки."""
    low, high = values[0], values[-1]
    if low == high:
        return []
    width = (high - low) / count
    return [low + width * index for index in range(1, count)]


def internal_edges(raw: Sequence[Decimal], values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    """Привести сырые границы к окончательному виду.

    §19.3: совпавшие границы удаляются, фактическое число бакетов фиксируется
    по тому, что осталось. Границы за пределами выборки тоже убираются — они
    создали бы заведомо пустой бакет.
    """
    return tuple(edge for edge in sorted(set(raw)) if values[0] < edge <= values[-1])
