"""Источник `core_payments` — ядро платежей.

Особенности схемы, важные для препроцессинга:
- время локальное наивное в формате `15.01.2026 14:30`, зона не указана
  и выводится из `branch_region` (§12, п.6);
- сумма — строка с пробелами-разделителями тысяч и запятой (§17.1);
- валюта в человеческом виде: `KZT`, `тенге`, `398` (§16);
- есть `loaded_at` (processing time), который §12.2 запрещает использовать
  вместо event time.

Сюда же инъектируется большая часть краевых случаев §29.2: этот источник
самый «человеческий», и грязь в нём выглядит естественно.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterator

from ..catalogs import (
    CORE_PAYMENT_OP_WEIGHTS,
    CURRENCY_ALIASES_TEXT,
    CURRENCY_WEIGHTS,
    DIRECTIONS,
    FX_BASE_RATE,
)
from ..clients import SyntheticClient
from ..config import GeneratorConfig
from ..edge_cases import Case, CaseLog, CasePlan, EventStream, has, planned_count
from ..records import RawRecord
from ..rng import derive_rng, weighted_choice
from ..timing import draw_local_datetime, format_naive_local, month_start, to_local, to_utc

SOURCE = "core_payments"
TIME_PATTERN = "%d.%m.%Y %H:%M"

# Порядок величины суммы по типу операции: exp(mu) — примерная медиана в тенге.
_AMOUNT_SHAPE: dict[str, tuple[float, float]] = {
    "TRF": (10.2, 1.1),
    "PMT": (9.2, 0.9),
    "LNP": (11.0, 0.5),
}

_EVENTS_PER_MONTH = 1.0


def generate(
    clients: list[SyntheticClient],
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    for client in clients:
        yield from _background(client, config, case_log)
        yield from _injections(client, config, plan, case_log)


# --------------------------------------------------------------------------- #
# Фоновые события
# --------------------------------------------------------------------------- #


def _background(
    client: SyntheticClient, config: GeneratorConfig, case_log: CaseLog
) -> Iterator[RawRecord]:
    rng = derive_rng(config.seed, SOURCE, client.client_id)
    active_from = client.active_from(config)
    if active_from > config.history_end:
        return

    months = max((config.history_end - active_from).days / 30.44, 0.5)
    natural = max(1, round(months * _EVENTS_PER_MONTH * client.activity * config.volume_scale))
    count = planned_count(client, EventStream.CORE_PAYMENTS, natural)

    if has(client, Case.NO_EVENTS):
        case_log.record(
            Case.NO_EVENTS,
            source=SOURCE,
            client_id=client.client_id,
            record_id="-",
            note="клиент без единого события во всех источниках",
        )
    if has(client, Case.LONG_HISTORY):
        case_log.record(
            Case.LONG_HISTORY,
            source=SOURCE,
            client_id=client.client_id,
            record_id="-",
            note=f"счёт с {client.account_open.isoformat()}, {count} платежей",
        )

    for index in range(count):
        moment_local = draw_local_datetime(rng, client.timezone, active_from, config.history_end)
        op_code = weighted_choice(rng, CORE_PAYMENT_OP_WEIGHTS)
        currency = weighted_choice(rng, CURRENCY_WEIGHTS)
        mu, sigma = _AMOUNT_SHAPE[op_code]
        amount_kzt = rng.lognormvariate(mu, sigma)
        amount = amount_kzt if currency == "KZT" else amount_kzt / FX_BASE_RATE[currency]

        record_id = f"CP-{client.client_ref}-{index:05d}"
        payload = _payload(
            record_id=record_id,
            client=client,
            moment_local=moment_local,
            op_code=op_code,
            amount=_format_amount(rng, amount),
            currency=rng.choice(CURRENCY_ALIASES_TEXT[currency]),
            direction="OUT" if op_code == "LNP" else weighted_choice(rng, DIRECTIONS),
            counterparty=_counterparty(rng),
            # Задержка загрузки — материал для late-arriving monitoring (§12.2).
            loaded_delay=timedelta(minutes=rng.randint(1, 4320)),
        )

        if has(client, Case.SINGLE_EVENT):
            case_log.record(
                Case.SINGLE_EVENT,
                source=SOURCE,
                client_id=client.client_id,
                record_id=record_id,
                note="единственное событие клиента за всю историю",
            )

        yield _record(client, payload, moment_local)


# --------------------------------------------------------------------------- #
# Краевые случаи
# --------------------------------------------------------------------------- #


def _injections(
    client: SyntheticClient,
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    if not client.cases or has(client, Case.NO_EVENTS):
        return

    rng = derive_rng(config.seed, SOURCE, "inject", client.client_id)
    emit = _Injector(client, plan, case_log, rng)

    yield from emit.dirty_values()
    yield from emit.out_of_range()
    yield from emit.duplicates()
    yield from emit.tie_breaks()
    yield from emit.fx_cases()
    yield from emit.historic()


class _Injector:
    """Помощник, чтобы каждая инъекция была в одну-две строки, а не в десять."""

    def __init__(
        self,
        client: SyntheticClient,
        plan: CasePlan,
        case_log: CaseLog,
        rng,
    ) -> None:
        self.client = client
        self.plan = plan
        self.log = case_log
        self.rng = rng

    def _emit(
        self,
        case: Case | None,
        tag: str,
        moment_utc: datetime,
        note: str,
        **overrides: Any,
    ) -> RawRecord:
        """Собрать запись с базовыми значениями и перекрыть нужные поля."""
        moment_local = to_local(moment_utc, self.client.timezone)
        record_id = f"CPX-{self.client.client_ref}-{tag}"
        payload = _payload(
            record_id=record_id,
            client=self.client,
            moment_local=moment_local,
            op_code="PMT",
            amount=_format_amount(self.rng, self.rng.lognormvariate(9.5, 0.6)),
            currency="KZT",
            direction="OUT",
            counterparty=_counterparty(self.rng),
        )
        payload.update(overrides)
        for key, value in list(payload.items()):
            if value is _ABSENT:
                del payload[key]

        if case is not None:
            self.log.record(
                case,
                source=SOURCE,
                client_id=self.client.client_id,
                record_id=record_id,
                note=note,
            )
        return _record(self.client, payload, moment_local)

    def dirty_values(self) -> Iterator[RawRecord]:
        cases = self.client.cases
        if Case.MISSING_NULL in cases:
            yield self._emit(Case.MISSING_NULL, "null-amount", self._moment(1), "amount = null", amount=None)
        if Case.MISSING_EMPTY in cases:
            yield self._emit(Case.MISSING_EMPTY, "empty-currency", self._moment(2), "currency = пустая строка", currency="")
        if Case.MISSING_PLACEHOLDER in cases:
            yield self._emit(Case.MISSING_PLACEHOLDER, "placeholder-direction", self._moment(3), "direction = 'N/A'", direction="N/A")
        if Case.MISSING_KEY in cases:
            yield self._emit(Case.MISSING_KEY, "no-direction", self._moment(4), "ключ direction отсутствует", direction=_ABSENT)
        if Case.NUMERIC_UNPARSABLE in cases:
            yield self._emit(Case.NUMERIC_UNPARSABLE, "bad-amount", self._moment(5), "amount = 'abc'", amount="abc")
        if Case.NUMERIC_NAN in cases:
            yield self._emit(Case.NUMERIC_NAN, "nan-amount", self._moment(6), "amount = 'NaN'", amount="NaN")
        if Case.NUMERIC_INF in cases:
            yield self._emit(Case.NUMERIC_INF, "inf-amount", self._moment(7), "amount = '-Inf'", amount="-Inf")
        if Case.UNKNOWN_EVENT_CODE in cases:
            yield self._emit(Case.UNKNOWN_EVENT_CODE, "unknown-op", self._moment(8), "op_code = 'XYZ' вне маппинга", op_code="XYZ")

    def out_of_range(self) -> Iterator[RawRecord]:
        cases = self.client.cases
        if Case.CLIP_LOW in cases:
            yield self._emit(Case.CLIP_LOW, "clip-low", self._moment(11), "сумма ниже TRAIN-диапазона", amount="0,01")
        if Case.CLIP_HIGH in cases:
            yield self._emit(Case.CLIP_HIGH, "clip-high", self._moment(12), "сумма выше TRAIN-диапазона", amount="5 000 000 000,00")
        if Case.FUTURE_TIMESTAMP in cases:
            yield self._emit(
                Case.FUTURE_TIMESTAMP,
                "future",
                self.plan.future_moment,
                "время далеко в будущем относительно периода истории",
            )

    def duplicates(self) -> Iterator[RawRecord]:
        cases = self.client.cases
        if Case.DUP_EXACT in cases:
            # Одна и та же строка приехала дважды: тот же record_id, тот же payload.
            record = self._emit(Case.DUP_EXACT, "dup-exact", self._moment(21), "полный дубль строки")
            yield record
            yield record

        if Case.DUP_BUSINESS in cases:
            moment = self._moment(22)
            shared = {"amount": "33 000,00", "currency": "KZT", "direction": "OUT", "counterparty_ref": "KZ700000000001"}
            yield self._emit(Case.DUP_BUSINESS, "dup-business-a", moment, "тот же бизнес-факт, другой source_record_id", **shared)
            yield self._emit(None, "dup-business-b", moment, "", **shared)

        if Case.DUP_CONFLICTING in cases:
            moment = self._moment(23)
            # Одинаковый ключ, разный payload — §9.3 запрещает решать это
            # случайным keep-first.
            conflicting = self._emit(Case.DUP_CONFLICTING, "dup-conflict", moment, "тот же source_record_id, разная сумма и версия", amount="10 000,00")
            yield conflicting
            yield self._emit(None, "dup-conflict", moment, "", amount="99 999,00", payload_version=2)

    def tie_breaks(self) -> Iterator[RawRecord]:
        cases = self.client.cases
        if Case.TIE_BREAK_SAME_SOURCE in cases:
            note = "два события одного источника с одинаковым timestamp"
            yield self._emit(Case.TIE_BREAK_SAME_SOURCE, "tie-a", self.plan.tie_break_moment, note)
            yield self._emit(None, "tie-b", self.plan.tie_break_moment, "")
        if Case.TIE_BREAK_CROSS_SOURCE in cases:
            yield self._emit(
                Case.TIE_BREAK_CROSS_SOURCE,
                "tie-cross",
                self.plan.tie_break_cross_moment,
                "парная запись к card_processing с тем же timestamp",
            )

    def fx_cases(self) -> Iterator[RawRecord]:
        cases = self.client.cases
        if Case.FX_FOREIGN in cases:
            yield self._emit(Case.FX_FOREIGN, "fx-usd", self.plan.fx_foreign_moment, "сумма в USD, курс на дату есть", amount="1 250,00", currency="USD")
        if Case.FX_FALLBACK in cases:
            yield self._emit(
                Case.FX_FALLBACK,
                "fx-weekend",
                self.plan.fx_fallback_moment,
                "воскресенье: курса нет, нужен последний прошлый",
                amount="980,50",
                currency="доллар",
            )
        if Case.FX_STALE_GAP in cases:
            yield self._emit(
                Case.FX_STALE_GAP,
                "fx-stale",
                self.plan.fx_stale_moment,
                f"курс не публиковался с {self.plan.fx_gap_start.isoformat()} — давность больше fx_max_staleness",
                amount="640,00",
                currency="EUR",
            )

    def historic(self) -> Iterator[RawRecord]:
        if Case.HISTORIC_TIMEZONE in self.client.cases:
            yield self._emit(
                Case.HISTORIC_TIMEZONE,
                "historic-tz",
                self.plan.historic_moment,
                "событие до 2024-03-01: локальный час считается по историческим правилам зоны",
            )

    def _moment(self, offset_days: int) -> datetime:
        """Спокойная дата внутри истории — чтобы инъекция не спорила
        с другими краевыми случаями по времени."""
        return self.plan.tie_break_moment - timedelta(days=offset_days)


class _Absent:
    """Маркер «ключа в payload быть не должно»."""


_ABSENT = _Absent()


# --------------------------------------------------------------------------- #
# Сборка записи
# --------------------------------------------------------------------------- #


def _payload(
    *,
    record_id: str,
    client: SyntheticClient,
    moment_local: datetime,
    op_code: str,
    amount: Any,
    currency: str,
    direction: Any,
    counterparty: str,
    loaded_delay: timedelta = timedelta(hours=2),
) -> dict[str, Any]:
    return {
        "source_record_id": record_id,
        "client_ref": client.client_ref,
        "branch_region": client.region,
        "operation_time": format_naive_local(moment_local, TIME_PATTERN),
        "op_code": op_code,
        "amount": amount,
        "currency": currency,
        "direction": direction,
        "counterparty_ref": counterparty,
        "payload_version": 1,
        "loaded_at": (to_utc(moment_local) + loaded_delay).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _record(client: SyntheticClient, payload: dict[str, Any], moment_local: datetime) -> RawRecord:
    return RawRecord(
        source=SOURCE,
        partition_date=month_start(to_utc(moment_local).date()),
        sort_key=str(payload["source_record_id"]),
        payload=payload,
    )


def _counterparty(rng) -> str:
    return f"KZ{rng.randrange(10**12):012d}"


def _format_amount(rng, value: float) -> str:
    """Сумма строкой. Часть записей приходит с разделителями тысяч
    (`15 000,50`), часть — без них (`15000.50`): обе формы реально встречаются
    в выгрузках, и NumericValidator обязан разобрать обе."""
    if rng.random() < 0.25:
        return f"{value:.2f}"
    integer, _, fraction = f"{value:,.2f}".partition(".")
    return f"{integer.replace(',', ' ')},{fraction}"
