"""DeterministicSampler — §19.6, §29 п.5.

Собирает выборку значений, по которой §19 потом считает границы бакетов.
Требование одно и жёсткое: **результат не зависит от числа воркеров**.

Классический reservoir sampling (алгоритм R) этому требованию не отвечает.
Он идёт по потоку и на каждом шаге бросает жребий из одного общего генератора,
поэтому результат зависит и от порядка записей, и от того, как поток порезали
между процессами. Собрать такие выборки из четырёх воркеров в одну, получив
тот же ответ, что у одного, невозможно в принципе.

Поэтому здесь **priority sampling**: каждая запись получает собственный ключ,
посчитанный только из неё самой и `global_seed` (§29.1 п.12), и в выборку
попадают `k` записей с наименьшим ключом. Свойство, ради которого всё
затевалось, становится арифметическим фактом: результат — это
`sorted(все ключи)[:k]`, а сортировка множества не зависит ни от порядка
поступления, ни от того, кто какую часть обработал.

Отсюда же и слияние воркеров: объединить две выборки — значит взять `k`
наименьших из объединения, что и делает `merge`.

§19.1: `MISSING` и невалидные значения в выборку не попадают — им нечего
делать в расчёте границ. Отсеиваются они здесь, а не в §19: к моменту fit
знать, почему значения нет, уже не из чего.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Iterator

from .core.hashing import reservoir_seed
from .core.monitor import DataQualityMonitor
from .feature_projection import ProjectedRecord
from .schema.constants import MISSING
from .schema.feature_schema import FeatureSchema, FieldType

COMPONENT = "deterministic_sampler"

SAMPLING_ALGORITHM: dict[str, Any] = {
    "name": "priority_sampling",
    "selects": "k_smallest_by_key",
    "key_source": "record_key_and_global_seed",
    "key_formula": "reservoir_seed(record_key, global_seed) — §29.1 п.12",
    "tie_break": "record_key",
    "record_key": "event_id, а для записи без события — ссылка на строку источника",
    "excludes": ["MISSING", "значения, не прошедшие §17"],
    "independent_of": ["число воркеров", "порядок записей", "порядок завершения задач"],
    "why_not_reservoir": (
        "классический алгоритм R идёт по потоку с одним общим генератором и "
        "зависит от нарезки потока; §29 п.5 требует обратного"
    ),
}
"""Описание алгоритма выборки — пункт перечня §31.

Артефакт словесный: §31 требует сохранить не выборку, а **алгоритм**, чтобы
другая реализация могла воспроизвести её из тех же данных. Формула ключа
живёт в `hash_policy` (§29.1 п.12) и здесь не дублируется — только названа.
"""

# Во сколько раз буфер может превысить размер выборки до уплотнения.
# Уплотнять на каждой записи — лишние сортировки, не уплотнять вовсе —
# держать в памяти весь поток.
_COMPACT_FACTOR = 4


@dataclass(frozen=True)
class SampledValue:
    """Значение, попавшее в выборку, вместе с ключом приоритета."""

    priority: int
    record_key: str
    value: Decimal

    def order(self) -> tuple[int, str]:
        """Ключ сортировки. `record_key` — вторым, чтобы совпадение
        приоритетов (невероятное, но возможное) не оставляло порядок
        на усмотрение реализации."""
        return (self.priority, self.record_key)


class FieldReservoir:
    """Выборка по одному полю: `k` значений с наименьшим ключом."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("размер выборки обязан быть положительным")
        self.size = size
        self.seen = 0
        self._buffer: list[SampledValue] = []

    def offer(self, item: SampledValue) -> None:
        self.seen += 1
        self._buffer.append(item)
        if len(self._buffer) > self.size * _COMPACT_FACTOR:
            self._compact()

    def merge(self, other: "FieldReservoir") -> None:
        """Присоединить выборку воркера.

        Отброшенные при уплотнении значения потерять нельзя: каждое из них
        больше всех `k` оставшихся, поэтому в итоговые `k` наименьших оно
        не попало бы ни при каком порядке слияния.
        """
        self.seen += other.seen
        self._buffer.extend(other._buffer)
        self._compact()

    def _compact(self) -> None:
        self._buffer = sorted(self._buffer, key=SampledValue.order)[: self.size]

    def selected(self) -> tuple[SampledValue, ...]:
        self._compact()
        return tuple(self._buffer)

    def values(self) -> tuple[Decimal, ...]:
        """Значения выборки в порядке возрастания — вход для fit границ §19."""
        return tuple(sorted(item.value for item in self.selected()))

    @property
    def truncated(self) -> bool:
        """Была ли выборка меньше потока. Если нет — «выборка» это все данные,
        и §19.6 фактически не применялся."""
        return self.seen > self.size


class DeterministicSampler:
    """Выборка значений числовых полей для расчёта границ бакетов."""

    def __init__(
        self,
        schema: FeatureSchema,
        *,
        sample_size: int,
        global_seed: int,
        monitor: DataQualityMonitor | None = None,
    ) -> None:
        self.schema = schema
        self.sample_size = sample_size
        self.global_seed = global_seed
        self._monitor = monitor
        self._fields = tuple(
            name
            for name, spec in schema.field_specs().items()
            if spec.type is FieldType.BUCKET
        )
        self.reservoirs: dict[str, FieldReservoir] = {
            name: FieldReservoir(sample_size) for name in self._fields
        }

    def consume(self, records: Iterable[ProjectedRecord]) -> Iterator[ProjectedRecord]:
        """Пропустить поток через себя, набирая выборку.

        Компонент ничего не меняет в записях: он наблюдатель на пути к §19,
        поэтому и стоит в цепочке прозрачно.
        """
        for record in records:
            self.offer(record)
            yield record

    def offer(self, record: ProjectedRecord) -> None:
        if not record.fields:
            return

        key = sample_key(record)
        priority = reservoir_seed(key, self.global_seed)

        for name in self._fields:
            value = record.fields.get(name)
            if value is None or value == MISSING:
                # §19.1: MISSING и невалидные в fit не участвуют.
                continue
            if not isinstance(value, Decimal):
                # До §17 значение ещё сырое; такой поток в fit не годится,
                # и молча пропустить его нельзя.
                raise TypeError(
                    f"{name}: в выборку пришло значение типа {type(value).__name__}; "
                    "сэмплер работает после NumericValidator (§17)"
                )
            self.reservoirs[name].offer(
                SampledValue(priority=priority, record_key=key, value=value)
            )

    def merge(self, other: "DeterministicSampler") -> None:
        for name, reservoir in other.reservoirs.items():
            self.reservoirs[name].merge(reservoir)

    def sample(self) -> dict[str, tuple[Decimal, ...]]:
        """Выборка по каждому полю — вход для fit границ §19."""
        return {name: self.reservoirs[name].values() for name in sorted(self.reservoirs)}

    def summary(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "by_field": {
                name: {
                    "seen": self.reservoirs[name].seen,
                    "selected": len(self.reservoirs[name].selected()),
                    "truncated": self.reservoirs[name].truncated,
                }
                for name in sorted(self.reservoirs)
            },
        }


def sample_key(record: ProjectedRecord) -> str:
    """Ключ записи для §29.1 п.12.

    У события это `event_id` — он стабилен и уникален по построению (§8).
    У записи без события (снимок профиля) — ссылка на строку источника: она
    тоже уникальна и не зависит ни от воркера, ни от порядка обработки.
    """
    return record.event_id or record.raw_reference
