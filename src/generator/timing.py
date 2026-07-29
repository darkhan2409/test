"""Работа со временем в генераторе.

Событие рождается как ЛОКАЛЬНОЕ время клиента (люди платят в обед и вечером,
а не «в 09:00 UTC»), и только потом переводится в UTC. Это даёт осмысленные
поведенческие hour/day, ради которых §12 и §25 требуют локальную IANA-зону.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc

# Суточный профиль активности: ночью почти пусто, пики в обед и вечером.
HOUR_WEIGHTS: tuple[float, ...] = (
    0.4, 0.2, 0.1, 0.1, 0.1, 0.3,   # 00..05
    0.8, 1.6, 2.6, 3.4, 3.8, 4.2,   # 06..11
    5.2, 4.8, 4.0, 3.8, 4.0, 4.6,   # 12..17
    5.4, 5.6, 4.8, 3.6, 2.4, 1.2,   # 18..23
)


def draw_local_datetime(
    rng: random.Random,
    tz_name: str,
    start: date,
    end: date,
) -> datetime:
    """Случайный момент в [start, end] как aware datetime в зоне клиента."""
    span_days = (end - start).days
    day = start + timedelta(days=rng.randint(0, max(span_days, 0)))
    hour = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
    naive = datetime(day.year, day.month, day.day, hour, rng.randrange(60), rng.randrange(60))
    return naive.replace(tzinfo=ZoneInfo(tz_name))


def to_utc(moment: datetime) -> datetime:
    """Перевести aware datetime в UTC."""
    return moment.astimezone(UTC)


def to_local(moment: datetime, tz_name: str) -> datetime:
    """Перевести aware datetime в локальную зону клиента."""
    return moment.astimezone(ZoneInfo(tz_name))


def next_weekday(start: date, weekday: int) -> date:
    """Первая дата не раньше `start` с заданным днём недели (0 = понедельник)."""
    shift = (weekday - start.weekday()) % 7
    return start + timedelta(days=shift)


def month_start(moment: date) -> date:
    """Начало месяца — имя партиции.

    Партиционируем помесячно, а не посуточно: ~30 партиций на источник дают
    достаточно параллелизма для проверки детерминизма на 4 воркерах и при этом
    не превращают `data/raw/` в тысячи файлов по несколько строк.
    """
    return date(moment.year, moment.month, 1)


def iter_months(start: date, end: date):
    """Начала всех месяцев, пересекающих период [start, end]."""
    current = month_start(start)
    last = month_start(end)
    while current <= last:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def iter_dates(start: date, end: date):
    """Все даты от start до end включительно."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def format_naive_local(moment: datetime, pattern: str) -> str:
    """Отформатировать локальное время БЕЗ смещения — так его отдают
    системы, которые про часовые пояса не знают (§12, п.6)."""
    return moment.strftime(pattern)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def epoch_millis(moment: datetime) -> int:
    """UTC-инстант в миллисекундах эпохи.

    Считается целочисленно через timedelta, а не через `timestamp() * 1000`:
    float-путь может дать разный результат на разных платформах, а нам нужна
    побайтовая воспроизводимость.
    """
    return (to_utc(moment) - _EPOCH) // timedelta(milliseconds=1)
