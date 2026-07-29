"""Синтетические клиенты.

У одного человека в каждой системе свой идентификатор — ровно та ситуация,
ради которой §7 требует IdentityResolver. Канонический `client_id` здесь
существует только как ground truth: препроцессинг обязан прийти к нему сам,
через versioned mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .catalogs import EMPLOYMENT_WEIGHTS, PRODUCTS, REGION_WEIGHTS, REGIONS
from .config import GeneratorConfig
from .rng import derive_rng, weighted_choice


@dataclass(frozen=True)
class SyntheticClient:
    """Один клиент со всеми своими идентификаторами и характером активности."""

    index: int
    client_id: str  # ground truth, в сырые источники не попадает
    client_ref: str  # core_payments
    cardholder_id: str  # card_processing
    login_id: str  # app_logs
    cif: str  # profile_snapshots
    device_id: str
    region: str
    timezone: str
    employment: str
    birth_year: int
    account_open: date
    activity: float  # множитель числа событий
    products: tuple[str, ...]
    salary: int
    # Метки краевых случаев §29.2; назначаются на шаге 0.3 плана.
    cases: frozenset[str] = frozenset()

    def active_from(self, config: GeneratorConfig) -> date:
        """Дата, раньше которой у клиента событий быть не может."""
        return max(config.history_start, self.account_open)

    def age_at(self, moment: date) -> int:
        return moment.year - self.birth_year


def build_clients(config: GeneratorConfig) -> list[SyntheticClient]:
    """Собрать всех клиентов. Каждый выводится из своего RNG-потока, поэтому
    состав не зависит от порядка обхода и от числа клиентов до него."""
    return [_build_client(config, index) for index in range(config.n_clients)]


def _build_client(config: GeneratorConfig, index: int) -> SyntheticClient:
    rng = derive_rng(config.seed, "client", index)

    region = weighted_choice(rng, REGION_WEIGHTS)
    employment = weighted_choice(rng, EMPLOYMENT_WEIGHTS)

    # Часть клиентов открывает счёт уже внутри периода истории — так появляются
    # «молодые» клиенты с короткой историей и корректным account_age на T.
    earliest_open = date(2008, 1, 1)
    latest_open = config.history_end - timedelta(days=30)
    account_open = earliest_open + timedelta(
        days=rng.randint(0, (latest_open - earliest_open).days)
    )

    activity = min(max(rng.lognormvariate(0.0, 0.6), 0.2), 4.0)

    product_count = rng.randint(1, 4)
    products = tuple(sorted(rng.sample(PRODUCTS, product_count)))

    return SyntheticClient(
        index=index,
        client_id=f"C{index:06d}",
        client_ref=f"{index:06d}",
        cardholder_id=f"CH{index:06d}",
        login_id=f"L{index:06d}",
        cif=f"CIF{index:06d}",
        device_id=f"DEV-{index:06d}-{rng.randrange(16**8):08x}",
        region=region,
        timezone=REGIONS[region],
        employment=employment,
        birth_year=rng.randint(1955, 2006),
        account_open=account_open,
        activity=activity,
        products=products,
        salary=int(min(max(rng.lognormvariate(12.7, 0.55), 90_000), 4_000_000)),
    )


def identity_mapping(clients: list[SyntheticClient]) -> dict[str, dict[str, str]]:
    """Ground truth «идентификатор источника → канонический client_id».

    Используется как заготовка для versioned identity_mapping препроцессинга
    (§7) и как эталон при проверке IdentityResolver.
    """
    mapping: dict[str, dict[str, str]] = {
        "core_payments.client_ref": {},
        "card_processing.cardholder_id": {},
        "app_logs.login_id": {},
        "profile_snapshots.cif": {},
    }
    for client in clients:
        mapping["core_payments.client_ref"][client.client_ref] = client.client_id
        mapping["card_processing.cardholder_id"][client.cardholder_id] = client.client_id
        mapping["app_logs.login_id"][client.login_id] = client.client_id
        mapping["profile_snapshots.cif"][client.cif] = client.client_id
    return mapping
