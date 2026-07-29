"""Источник `fx_rates` — исторические курсы валют к тенге.

Курсы публикуются только по рабочим дням. Это не дефект синтетики, а
обязательное условие: §18.2 требует при отсутствии курса на дату события брать
последний доступный НЕ ПОЗЖЕ события, и без выходных этот путь никогда бы
не выполнился.

Дополнительно в календаре есть искусственный разрыв публикации (§29.2): в нём
последний доступный курс оказывается старше `fx_max_staleness`, и сумма обязана
получить MISSING, а не пересчитаться по чему попало.

Курс — случайное блуждание от базового значения, поэтому суммы за 2023 и за
2026 год конвертируются по разным курсам, как и должно быть при исторической
FX-нормализации.
"""

from __future__ import annotations

from typing import Iterator

from ..catalogs import FX_BASE_RATE
from ..config import GeneratorConfig
from ..edge_cases import Case, CaseLog, CasePlan
from ..records import RawRecord
from ..rng import derive_rng
from ..timing import iter_dates, month_start

SOURCE = "fx_rates"
RATE_SOURCE = "NBK"

_SATURDAY = 5


def generate(
    config: GeneratorConfig, plan: CasePlan, case_log: CaseLog
) -> Iterator[RawRecord]:
    rng = derive_rng(config.seed, SOURCE)
    rates = dict(FX_BASE_RATE)
    gap = plan.fx_gap_dates

    case_log.record(
        Case.FX_STALE_GAP,
        source=SOURCE,
        client_id="-",
        record_id="-",
        note=(
            f"курсы не публикуются {plan.fx_gap_days} дней подряд с "
            f"{plan.fx_gap_start.isoformat()}"
        ),
    )

    for rate_date in iter_dates(config.history_start, config.history_end):
        # Блуждание считаем каждый день, чтобы выходные и разрыв не «замораживали»
        # курс, а публикуем только по рабочим дням вне разрыва.
        for currency in sorted(rates):
            rates[currency] = max(rates[currency] * (1.0 + rng.gauss(0.0, 0.004)), 1.0)

        if rate_date.weekday() >= _SATURDAY or rate_date in gap:
            continue

        for currency in sorted(rates):
            payload = {
                "rate_date": rate_date.isoformat(),
                "currency": currency,
                "rate_to_kzt": f"{rates[currency]:.4f}",
                "source": RATE_SOURCE,
            }
            yield RawRecord(
                source=SOURCE,
                partition_date=month_start(rate_date),
                sort_key=f"{rate_date.isoformat()}|{currency}",
                payload=payload,
            )
