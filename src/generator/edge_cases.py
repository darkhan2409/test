"""Краевые случаи §29.2: какие бывают, кому назначены и когда происходят.

Случайная генерация краевые случаи не гарантирует: на маленьком golden-наборе
дубликат или событие ровно в T могут просто не выпасть. Поэтому каждый случай
закреплён за конкретным клиентом и инъектируется адресно, а `CaseLog`
записывает, куда именно он попал — потом по этому манифесту проверки бьют
точечно, а не ищут иголку в 76 тысячах строк.

Роли клиентов фиксированы по индексу: клиент 0 всегда «длинная история»,
клиент 7 всегда «дубликаты». Это делает состав случаев воспроизводимым и
одинаковым в main- и golden-наборах.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum

from .clients import SyntheticClient
from .config import GeneratorConfig
from .timing import next_weekday

UTC = timezone.utc

MONDAY = 0
SUNDAY = 6


class Case(StrEnum):
    """Перечень краевых случаев. Каждый обязан встретиться в датасете."""

    # Форма истории клиента
    LONG_HISTORY = "long_history"  # история длиннее model window → WINDOW_START
    SINGLE_EVENT = "single_event"  # ровно одно событие за жизнь → lifetime_first
    NO_PROFILE = "no_profile"  # нет ни одного снимка профиля (§6, п.3)
    NO_EVENTS = "no_events"  # профиль есть, событий нет
    HISTORIC_TIMEZONE = "historic_timezone"  # события до 2024-03-01 в зоне UTC+6

    # Пропуски и невалидные значения (§15, §17)
    MISSING_NULL = "missing_null"
    MISSING_EMPTY = "missing_empty"
    MISSING_PLACEHOLDER = "missing_placeholder"
    MISSING_KEY = "missing_key"
    NUMERIC_UNPARSABLE = "numeric_unparsable"
    NUMERIC_NAN = "numeric_nan"
    NUMERIC_INF = "numeric_inf"
    NEGATIVE_AGE = "negative_age"  # отрицательное там, где оно невалидно (§17.3)
    UNKNOWN_EVENT_CODE = "unknown_event_code"  # неизвестный код → quarantine (§10)

    # Числовой диапазон и время
    CLIP_LOW = "clip_low"  # ниже min_train_edge (§19.4)
    CLIP_HIGH = "clip_high"  # выше max_train_edge
    FUTURE_TIMESTAMP = "future_timestamp"  # время из будущего (§12, п.9)

    # Дедупликация (§9)
    DUP_EXACT = "dup_exact"
    DUP_BUSINESS = "dup_business"
    DUP_CONFLICTING = "dup_conflicting"

    # Детерминированный порядок (§13)
    TIE_BREAK_SAME_SOURCE = "tie_break_same_source"
    TIE_BREAK_CROSS_SOURCE = "tie_break_cross_source"

    # FX (§18)
    FX_FOREIGN = "fx_foreign"
    FX_FALLBACK = "fx_fallback"  # курса на дату нет → берём последний прошлый
    FX_STALE_GAP = "fx_stale_gap"  # разрыв больше fx_max_staleness → MISSING

    # Cutoff (§14)
    BOUNDARY_AT_T = "boundary_at_t"  # ровно T — входит
    BOUNDARY_AFTER_T = "boundary_after_t"  # T + 1 секунда — не входит
    SESSION_CROSSING_T = "session_crossing_t"  # сессия начата до T, продолжена после

    # Словарь токенайзера
    RARE_VALUE = "rare_value"  # реже min_count → RARE
    UNSEEN_VALUE = "unseen_value"  # только после T → [UNK]

    # Прочее
    MULTIVALUE_OVERFLOW = "multivalue_overflow"  # длиннее max_values_per_field (§21)
    NAIVE_TIMESTAMP = "naive_timestamp"  # время без зоны (§12, п.6)


class EventStream(StrEnum):
    """Потоки фоновых событий, объём которых зависит от роли клиента."""

    CORE_PAYMENTS = "core_payments"
    CARD_PROCESSING = "card_processing"
    APP_SESSIONS = "app_logs.sessions"
    APP_PUSH = "app_logs.push"


# Роль клиента по индексу. Один клиент несёт несколько родственных случаев,
# чтобы golden-набор оставался маленьким.
ROLE_CASES: tuple[frozenset[Case], ...] = (
    frozenset({Case.LONG_HISTORY}),
    frozenset({Case.SINGLE_EVENT}),
    frozenset({Case.NO_PROFILE}),
    frozenset({Case.NO_EVENTS}),
    frozenset({Case.HISTORIC_TIMEZONE, Case.MULTIVALUE_OVERFLOW}),
    frozenset(
        {
            Case.MISSING_NULL,
            Case.MISSING_EMPTY,
            Case.MISSING_PLACEHOLDER,
            Case.MISSING_KEY,
            Case.NUMERIC_UNPARSABLE,
            Case.NUMERIC_NAN,
            Case.NUMERIC_INF,
            Case.UNKNOWN_EVENT_CODE,
            Case.NEGATIVE_AGE,
        }
    ),
    frozenset({Case.CLIP_LOW, Case.CLIP_HIGH, Case.FUTURE_TIMESTAMP}),
    frozenset({Case.DUP_EXACT, Case.DUP_BUSINESS, Case.DUP_CONFLICTING}),
    frozenset({Case.TIE_BREAK_SAME_SOURCE, Case.TIE_BREAK_CROSS_SOURCE}),
    frozenset({Case.FX_FOREIGN, Case.FX_FALLBACK, Case.FX_STALE_GAP}),
    frozenset({Case.BOUNDARY_AT_T, Case.BOUNDARY_AFTER_T, Case.SESSION_CROSSING_T}),
    frozenset({Case.RARE_VALUE, Case.UNSEEN_VALUE}),
)

MIN_CLIENTS = len(ROLE_CASES)

# Клиент с историческими событиями должен жить в зоне, которая ДО 2024-03-01
# отличалась от нынешнего UTC+05, иначе случай ничего не проверяет.
_HISTORIC_REGION = "ALMATY"
_HISTORIC_TIMEZONE = "Asia/Almaty"

_LONG_HISTORY_MULTIPLIER = 4.0
_RARE_VALUE_COUNT = 3
_UNSEEN_VALUE_COUNT = 2


@dataclass(frozen=True)
class CasePlan:
    """Когда именно происходят краевые случаи. Всё выводится из конфига,
    поэтому один и тот же конфиг даёт одни и те же даты."""

    boundary_at_t: datetime
    boundary_after_t: datetime
    tie_break_moment: datetime  # пара записей одного источника
    tie_break_cross_moment: datetime  # пара записей разных источников
    historic_moment: datetime
    future_moment: datetime
    fx_foreign_moment: datetime
    fx_fallback_moment: datetime
    fx_stale_moment: datetime
    fx_gap_start: date
    fx_gap_days: int
    unseen_from: date
    rare_value_count: int = _RARE_VALUE_COUNT
    unseen_value_count: int = _UNSEEN_VALUE_COUNT

    @property
    def fx_gap_dates(self) -> frozenset[date]:
        """Даты, в которые курс не публикуется вовсе."""
        return frozenset(
            self.fx_gap_start + timedelta(days=offset) for offset in range(self.fx_gap_days)
        )


def build_plan(config: GeneratorConfig) -> CasePlan:
    """Рассчитать календарь краевых случаев."""
    cutoff = config.cutoff_time.astimezone(UTC)

    # Воскресенье: курса в этот день нет, ближайший прошлый — пятничный,
    # давность 2 дня < fx_max_staleness, поэтому fallback обязан сработать.
    fx_fallback_date = next_weekday(date(2025, 6, 1), SUNDAY)

    # Разрыв публикации на 8 дней: последний доступный курс окажется старше
    # трёх суток, и §18.2 требует MISSING, а не подстановку чего попало.
    fx_gap_start = next_weekday(date(2025, 7, 1), MONDAY)
    fx_gap_days = 8
    fx_stale_date = fx_gap_start + timedelta(days=fx_gap_days - 1)

    return CasePlan(
        boundary_at_t=cutoff,
        boundary_after_t=cutoff + timedelta(seconds=1),
        # Секунды нулевые: core_payments пишет время с точностью до минуты,
        # и без этого «одинаковый timestamp» у двух источников не совпал бы.
        tie_break_moment=datetime(2025, 6, 10, 9, 0, 0, tzinfo=UTC),
        tie_break_cross_moment=datetime(2025, 6, 10, 10, 0, 0, tzinfo=UTC),
        historic_moment=datetime(2024, 1, 18, 8, 30, 0, tzinfo=UTC),
        future_moment=datetime(config.history_end.year + 2, 5, 6, 7, 8, 0, tzinfo=UTC),
        fx_foreign_moment=datetime(2025, 9, 17, 11, 15, 0, tzinfo=UTC),
        fx_fallback_moment=datetime(
            fx_fallback_date.year, fx_fallback_date.month, fx_fallback_date.day, 10, 0, tzinfo=UTC
        ),
        fx_stale_moment=datetime(
            fx_stale_date.year, fx_stale_date.month, fx_stale_date.day, 12, 0, tzinfo=UTC
        ),
        fx_gap_start=fx_gap_start,
        fx_gap_days=fx_gap_days,
        unseen_from=cutoff.date() + timedelta(days=5),
    )


def assign_cases(
    clients: list[SyntheticClient], config: GeneratorConfig
) -> list[SyntheticClient]:
    """Раздать роли первым `len(ROLE_CASES)` клиентам.

    Некоторым ролям мало пометки: клиенту с историческими событиями нужна
    «правильная» зона, а клиенту с длинной историей — рано открытый счёт.
    """
    if len(clients) < MIN_CLIENTS:
        raise ValueError(
            f"нужно минимум {MIN_CLIENTS} клиентов, чтобы разместить все краевые случаи, "
            f"передано {len(clients)}"
        )

    updated = list(clients)
    for index, cases in enumerate(ROLE_CASES):
        updated[index] = _apply_role(updated[index], cases, config)
    return updated


def _apply_role(
    client: SyntheticClient, cases: frozenset[Case], config: GeneratorConfig
) -> SyntheticClient:
    changes: dict[str, object] = {"cases": cases}

    if Case.HISTORIC_TIMEZONE in cases:
        changes["region"] = _HISTORIC_REGION
        changes["timezone"] = _HISTORIC_TIMEZONE
        changes["account_open"] = min(client.account_open, config.history_start)

    if Case.LONG_HISTORY in cases:
        changes["account_open"] = min(client.account_open, date(2010, 1, 1))
        changes["activity"] = client.activity * _LONG_HISTORY_MULTIPLIER

    return replace(client, **changes)


def has(client: SyntheticClient, case: Case) -> bool:
    return case in client.cases


def planned_count(client: SyntheticClient, stream: EventStream, natural: int) -> int:
    """Скорректировать число фоновых событий под роль клиента."""
    if has(client, Case.NO_EVENTS):
        return 0
    if has(client, Case.SINGLE_EVENT):
        # Единственное событие за всю жизнь кладём в core_payments.
        return 1 if stream is EventStream.CORE_PAYMENTS else 0
    return natural


@dataclass(frozen=True)
class CaseEntry:
    """Куда попал конкретный краевой случай."""

    case: str
    source: str
    client_id: str
    record_id: str
    note: str


class CaseLog:
    """Собирает манифест краевых случаев по ходу генерации."""

    def __init__(self) -> None:
        self._entries: list[CaseEntry] = []

    def record(
        self,
        case: Case,
        *,
        source: str,
        client_id: str,
        record_id: str,
        note: str = "",
    ) -> None:
        self._entries.append(
            CaseEntry(
                case=str(case),
                source=source,
                client_id=client_id,
                record_id=record_id,
                note=note,
            )
        )

    def missing_cases(self) -> list[str]:
        """Случаи, которых в датасете не оказалось.

        Пустой список — обязательное условие: инъекция, которая молча не
        сработала, хуже отсутствующей, потому что создаёт ложную уверенность.
        """
        present = {entry.case for entry in self._entries}
        return sorted(str(case) for case in Case if str(case) not in present)

    def as_manifest(self) -> dict[str, list[dict[str, str]]]:
        manifest: dict[str, list[dict[str, str]]] = {}
        for entry in sorted(
            self._entries, key=lambda item: (item.case, item.source, item.record_id)
        ):
            manifest.setdefault(entry.case, []).append(
                {
                    "source": entry.source,
                    "client_id": entry.client_id,
                    "record_id": entry.record_id,
                    "note": entry.note,
                }
            )
        return manifest
