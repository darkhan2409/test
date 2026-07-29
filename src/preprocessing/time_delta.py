"""Time delta edges — §25.2, §27 шаг 13.

Отдельный компонент, а не поле бакетизации §19: §25.2 говорит об этом прямо,
и различие не косметическое. Границы полей события применяются здесь, на
препроцессинге, а границы дельты — токенайзером, **после** truncation окна
(§25.1). Значение дельты до обрезки и после — разные числа, поэтому дельта не
может быть посчитана как обычное поле и заморожена вместе с ними.

**Дельта считается, но не сохраняется.** `TimeFeatureBuilder` (2.17) её не
считает вовсе — поля, которого нет, нельзя случайно отдать модели. Здесь она
нужна ровно для подбора границ и живёт внутри расчёта: `TimeDeltaEdges` несёт
границы, а не значения.

**Первое событие клиента дельты не имеет** и в выборку не идёт. Это тот же
принцип, что §19.1: значение, которого нет, границы не образует. Подставить
ему ноль означало бы сказать «предыдущее событие было в тот же миг» — и
сдвинуть первый бакет на всех клиентах сразу.

**`FIRST_EVENT` и `WINDOW_START` — вне границ.** §25.2 и §10.2 токенайзера
называют их зарезервированными значениями delta-канала «вне обычных
`time_delta_edges`». Поэтому они публикуются отдельным ключом артефакта и в
`labels()` не входят: попасть в них расчётом нельзя, они не бакеты.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import reservoir_seed
from .core.quantiles import internal_edges, quantile_edges
from .feature_projection import ProjectedRecord
from .sampler import FieldReservoir, SampledValue

COMPONENT = "time_delta"
LABEL_PREFIX = "delta_bucket_"

_ONE_SECOND = timedelta(seconds=1)

FIRST_EVENT = "FIRST_EVENT"
WINDOW_START = "WINDOW_START"
RESERVED = (FIRST_EVENT, WINDOW_START)
"""Зарезервированные значения delta-канала (§25.2, tokenizer §10.2).

`FIRST_EVENT` — у самого раннего известного события клиента, `WINDOW_START` —
у первого события внутри окна, когда история до него обрезана. Различить их
токенайзер может только по `lifetime_first` (§25.1), который проставляет 2.17.
"""

UNIT = "seconds"
"""Единица дельты. Секунды, а не минуты и не часы: разрешение задаётся один
раз и попадает в хэш состояния, а перевод в другую единицу сдвинул бы все
границы, не изменив ни одного значения."""


class TimeDeltaError(RuntimeError):
    """Ошибка расчёта границ дельты — блокирующая, до заморозки артефакта."""


class DeltaMethod(StrEnum):
    """Метод подбора границ дельты.

    Значение одно. Дельты между событиями распределены с тяжёлым хвостом —
    минуты у активного клиента и месяцы у спящего, — и равная ширина отдала бы
    почти всё в первый бакет. Объявить нереализованный метод нечем: ветки
    расчёта под него не существует.
    """

    QUANTILE = "quantile"


class TimeDeltaConfig(BaseModel):
    """Версионируемая конфигурация delta-канала (§25.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_delta_edges_version: str = Field(min_length=1)
    method: DeltaMethod = DeltaMethod.QUANTILE
    bucket_count: int = Field(gt=1)
    sample_size: int = Field(gt=0)

    def state(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_time_delta_config(path: Path) -> TimeDeltaConfig:
    """Загрузить конфиг delta-канала."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TimeDeltaError(f"{path}: ожидался YAML-объект")
    return TimeDeltaConfig.model_validate(document)


@dataclass(frozen=True)
class TimeDeltaEdges:
    """Замороженный артефакт BUILD: границы delta-канала (§25.2)."""

    version: str
    method: DeltaMethod
    requested_count: int
    edges: tuple[Decimal, ...]
    """Внутренние границы в секундах, по возрастанию. Бакетов `len(edges) + 1`."""

    min_train: Decimal
    max_train: Decimal
    sample_size: int
    deltas_seen: int

    @property
    def bucket_count(self) -> int:
        """Фактическое число бакетов — §19.3 фиксирует именно его."""
        return len(self.edges) + 1

    def labels(self) -> tuple[str, ...]:
        """Только обычные бакеты. Зарезервированные значения сюда не входят —
        см. `RESERVED` и §25.2."""
        return tuple(f"{LABEL_PREFIX}{index}" for index in range(self.bucket_count))

    def domain(self) -> tuple[str, ...]:
        """Всё, что может оказаться в delta-канале: бакеты плюс зарезервированные.

        Токенайзеру нужен полный whitelist (§2.2), иначе редкое
        зарезервированное значение схлопнулось бы в `RARE` по `min_count`.
        Порядок фиксирован: сначала бакеты, потом зарезервированные — от него
        зависят id в словаре.
        """
        return self.labels() + RESERVED

    def intervals(self) -> tuple[dict[str, str | None], ...]:
        """Границы бакетов для decode и observability (§2.1).

        Крайние интервалы открыты: дельта меньше `min_train` возможна, и
        закрывать интервал снизу значило бы обещать, что таких не бывает.
        """
        bounds: list[dict[str, str | None]] = []
        for index in range(self.bucket_count):
            low = str(self.edges[index - 1]) if index > 0 else None
            high = str(self.edges[index]) if index < len(self.edges) else None
            bounds.append({"label": f"{LABEL_PREFIX}{index}", "low": low, "high": high})
        return tuple(bounds)

    def state(self) -> dict[str, Any]:
        """Состояние для §29.1 и §30.

        Границы — строки десятичных дробей по той же причине, что и в §19:
        дельта, попавшая ровно на границу, обязана всегда попадать в один и
        тот же бакет, а двоичная дробь этого не гарантирует.
        """
        return {
            "time_delta_edges_version": self.version,
            "method": str(self.method),
            "unit": UNIT,
            "requested_bucket_count": self.requested_count,
            "bucket_count": self.bucket_count,
            "edges": [str(edge) for edge in self.edges],
            "min_train_delta": str(self.min_train),
            "max_train_delta": str(self.max_train),
            "sample_size": self.sample_size,
            "deltas_seen": self.deltas_seen,
            "labels": list(self.labels()),
            "reserved": list(RESERVED),
            "domain": list(self.domain()),
            "intervals": [dict(item) for item in self.intervals()],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TimeDeltaEdges":
        """Загрузить замороженные границы (§28 п.1).

        `labels`, `reserved`, `domain` и `intervals` в артефакте есть, но не
        читаются: они выводятся из границ, и чтение сделало бы их вторым
        источником истины. `reserved` вдобавок задан кодом (§25.2) — прочитать
        его из файла значило бы дать файлу право переименовать
        `FIRST_EVENT`.
        """
        return cls(
            version=str(state["time_delta_edges_version"]),
            method=DeltaMethod(state["method"]),
            requested_count=int(state["requested_bucket_count"]),
            edges=tuple(Decimal(item) for item in state["edges"]),
            min_train=Decimal(state["min_train_delta"]),
            max_train=Decimal(state["max_train_delta"]),
            sample_size=int(state["sample_size"]),
            deltas_seen=int(state["deltas_seen"]),
        )


class DeltaSampler:
    """Детерминированная выборка дельт (§19.6, §29 п.5).

    Тот же priority sampling, что у значений полей, и намеренно тот же код:
    два разных алгоритма выборки в одном пайплайне означали бы два разных
    ответа на вопрос «зависит ли результат от нарезки потока».

    Ключ записи — `event_id` **последующего** события пары. Он стабилен (§8) и
    принадлежит ровно одной паре: у первого события клиента дельты нет, так
    что двух пар с одним ключом не бывает.
    """

    def __init__(self, *, sample_size: int, global_seed: int) -> None:
        self.global_seed = global_seed
        self.reservoir = FieldReservoir(sample_size)

    def offer(self, *, event_id: str, delta: Decimal) -> None:
        if delta < 0:
            # Отрицательная дельта означает, что timeline не отсортирован
            # (§13). Молча взять модуль значило бы посчитать границы по
            # выдуманным данным.
            raise TimeDeltaError(
                f"{event_id}: отрицательная дельта {delta} — timeline пришёл "
                "неупорядоченным (§13)"
            )
        self.reservoir.offer(
            SampledValue(
                priority=reservoir_seed(event_id, self.global_seed),
                record_key=event_id,
                value=delta,
            )
        )

    def merge(self, other: "DeltaSampler") -> None:
        self.reservoir.merge(other.reservoir)

    def values(self) -> tuple[Decimal, ...]:
        return self.reservoir.values()

    @property
    def seen(self) -> int:
        return self.reservoir.seen


def collect_deltas(records: Iterable[ProjectedRecord], sampler: DeltaSampler) -> None:
    """Пройти по timeline и предложить выборке дельты соседних событий.

    Записи обязаны идти в порядке §13 и подряд по клиенту — именно так их
    отдаёт `TimelineBuilder`. Проверяется не порядок (его гарантирует §13), а
    то, что дельта берётся внутри одного клиента: пара из событий разных
    клиентов не значит ничего.
    """
    previous_client: str | None = None
    previous_time: datetime | None = None

    for record in records:
        event_type = getattr(record, "event_type", None)
        moment = getattr(record, "timestamp_utc", None)
        if event_type is None or moment is None:
            # Снимок профиля привязан к T целиком, дельты у него нет.
            continue

        client = record.client_id
        if client != previous_client:
            previous_client, previous_time = client, moment
            continue

        if previous_time is None:
            raise TimeDeltaError(f"{record.raw_reference}: событие без времени в timeline")

        event_id = record.event_id
        if not event_id:
            raise TimeDeltaError(
                f"{record.raw_reference}: событие без event_id — дельту нечем "
                "привязать к паре (§8)"
            )
        sampler.offer(
            event_id=event_id,
            delta=Decimal((moment - previous_time) // _ONE_SECOND),
        )
        previous_time = moment


def fit_time_delta_edges(sampler: DeltaSampler, config: TimeDeltaConfig) -> TimeDeltaEdges:
    """Посчитать границы delta-канала по TRAIN-выборке (§25.2, §27 шаг 13)."""
    values = list(sampler.values())
    if not values:
        # Ни одной пары соседних событий: либо у каждого клиента ровно одно
        # событие, либо timeline не дошёл. Молча выдать один бакет — спрятать
        # это от читателя артефакта.
        raise TimeDeltaError(
            "в TRAIN нет ни одной дельты: у delta-канала не из чего считать границы"
        )

    raw = quantile_edges(values, config.bucket_count)
    return TimeDeltaEdges(
        version=config.time_delta_edges_version,
        method=config.method,
        requested_count=config.bucket_count,
        edges=internal_edges(raw, values),
        min_train=values[0],
        max_train=values[-1],
        sample_size=len(values),
        deltas_seen=sampler.seen,
    )
