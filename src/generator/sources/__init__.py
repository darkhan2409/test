"""Генераторы сырых источников.

Схемы источников намеренно разные: у каждого свой формат времени, свой способ
записать сумму и свой идентификатор клиента. Именно это разнообразие проверяет
Source Contract (§4), IdentityResolver (§7) и TimestampNormalizer (§12).

`fx_rates` не привязан к клиентам, поэтому его `generate` принимает только
конфиг — вызывающий код учитывает это явно.
"""

from . import app_logs, card_processing, core_payments, fx_rates, profile_snapshots

SOURCE_NAMES = (
    core_payments.SOURCE,
    card_processing.SOURCE,
    app_logs.SOURCE,
    profile_snapshots.SOURCE,
    fx_rates.SOURCE,
)

__all__ = [
    "SOURCE_NAMES",
    "app_logs",
    "card_processing",
    "core_payments",
    "fx_rates",
    "profile_snapshots",
]
