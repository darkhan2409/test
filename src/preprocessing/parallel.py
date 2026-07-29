"""Параллельная обработка и доказательство, что она состоялась — §29, §27 шаг 16.

§29 п.10 требует сравнить выход одного воркера с выходом нескольких. Само по
себе совпадение ничего не доказывает: если работа фактически не разделилась —
все партиции достались одному процессу, или пул отработал их в каноническом
порядке — сравнение сравнивает прогон с самим собой и всегда зелёное.

Поэтому здесь два разных факта, и путать их нельзя:

- **выходы совпали** — это `ParallelOutputMismatchError`, если нет;
- **параллельность состоялась** — это `ParallelismNotProvenError`, если нет.

Второе проверяется по следам, которые оставляет сам прогон: идентификаторы
процессов, фактическое распределение партиций между ними и порядок
завершения. Вырожденное распределение или порядок завершения, совпавший с
каноническим, означают не «проверка пройдена», а **«проверка не проведена»**.

**Что именно идёт параллельно.** Партиции читаются и разрешаются по клиенту
(§4, §7) в отдельных процессах; дальше цепочка сходится. Это не ограничение
реализации, а свойство §37.2: дедупликация собирает поток целиком (дубль
может лежать в другой партиции), индекс зон строится по всем профилям,
сессионизация требует порядка внутри клиента. §29 п.3 это и предполагает,
требуя строгого порядка слияния. Заявляется ровно столько, сколько сделано.

Слияние идёт **по каноническому порядку партиций**, а не по порядку
завершения задач (§29 пп. 1–3).

Честная оговорка о том, чем детерминизм держится на самом деле: мутация
«слить по порядку завершения» выживает — и unit-тесты, и полный прогон 4.1
остаются зелёными. Так и должно быть, и это факт о цепочке, а не дыра в
проверке. Дедупликация выдаёт записи в каноническом порядке
(`sorted(survivors, key=_canonical_position)`), TimelineBuilder сортирует
внутри клиента, сэмплер порядконезависим по построению — всё, что ниже,
пересортировывает само. Канонический порядок слияния остаётся, потому что
его требует §29 п.3 и потому что он страхует **будущие** компоненты: полагаться
на то, что каждый следующий не забудет отсортироваться, — не то же самое,
что не зависеть от порядка.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.monitor import DataQualityMonitor
from .core.quarantine import Quarantine
from .identity_resolver import IdentityResolver, load_identity_mapping
from .records import IdentifiedRecord
from .schema import load_source_contracts
from .source_reader import Partition, SourceReader

COMPONENT = "parallel"


class ParallelismNotProvenError(RuntimeError):
    """Параллельность не доказана — проверка считается **непроведённой**.

    Намеренно отдельный тип: «выходы совпали, но работа не разделилась» и
    «выходы разошлись» — разные факты, и один не должен маскироваться под
    другой в отчёте.
    """


class ParallelOutputMismatchError(RuntimeError):
    """Выход multi-worker отличается от single-worker — §29, блокирующая."""


@dataclass(frozen=True)
class ParallelEvidence:
    """Следы прогона, по которым видно, что параллельность состоялась."""

    parent_pid: int
    requested_workers: int
    by_pid: dict[int, tuple[str, ...]]
    """Какие партиции достались какому процессу. Ключ — реальный PID."""

    completion_order: tuple[str, ...]
    """Партиции в порядке фактического завершения задач."""

    canonical_order: tuple[str, ...]
    """Партиции в порядке §29 п.1 — тот, в котором идёт слияние."""

    def summary(self) -> dict[str, Any]:
        return {
            "parent_pid": self.parent_pid,
            "requested_workers": self.requested_workers,
            "actual_processes": len(self.by_pid),
            "partitions_by_pid": {
                str(pid): len(items) for pid, items in sorted(self.by_pid.items())
            },
            "completion_order_differs_from_canonical": (
                self.completion_order != self.canonical_order
            ),
            "partitions": len(self.canonical_order),
        }

    def degeneracies(self) -> list[str]:
        """Причины, по которым сравнение выходов ничего не доказывает."""
        problems: list[str] = []

        if len(self.by_pid) < 2:
            only = next(iter(self.by_pid), None)
            problems.append(
                f"работа не разделилась: все {len(self.canonical_order)} партиций "
                f"обработал один процесс (pid {only})"
            )

        if self.parent_pid in self.by_pid:
            problems.append(
                f"партиции обработаны родительским процессом (pid {self.parent_pid}) — "
                "пул не запустился, и параллельности не было"
            )

        if self.completion_order == self.canonical_order:
            problems.append(
                "порядок завершения задач совпал с каноническим — прогон не проверил "
                "независимость слияния от порядка завершения (§29 пп. 3, 6)"
            )

        return problems


def require_parallelism(evidence: ParallelEvidence) -> None:
    """Убедиться, что сравнивать вообще было что.

    Ничего не возвращает: единственный исход, кроме молчаливого успеха, —
    исключение. Вердикта «параллельность слабая, но засчитаем» не существует.
    """
    problems = evidence.degeneracies()
    if problems:
        raise ParallelismNotProvenError(
            "проверка §29 п.10 не проведена, а не пройдена — "
            + "; ".join(problems)
            + f". Следы прогона: {evidence.summary()}"
        )


# --------------------------------------------------------------------------- #
# Воркер
# --------------------------------------------------------------------------- #

_WORKER: dict[str, Any] = {}


def _init_worker(config_dir: str, raw_dir: str) -> None:
    """Загрузить контракты один раз на процесс.

    Загрузка внутри воркера, а не передача объектов из родителя, — это ещё и
    проверка: разбор конфигов обязан быть детерминированным, иначе четыре
    процесса получили бы четыре немного разных контракта.
    """
    registry = load_source_contracts(Path(config_dir) / "source_contracts.yaml")
    mapping = load_identity_mapping(Path(config_dir) / "identity_mapping.yaml", registry)
    _WORKER["registry"] = registry
    _WORKER["mapping"] = mapping
    _WORKER["raw_dir"] = Path(raw_dir)


def _read_partition(
    canonical_path: str, source: str, processing_time: datetime, version: str
) -> dict[str, Any]:
    """Прочитать одну партицию и разрешить `client_id` (§4, §7).

    Источник передаётся родителем, а не выводится из пути. Вывод по первому
    сегменту работал бы ровно до первой партиции глубже одного уровня, и
    сломался бы только в параллельном пути — то есть там, где это заметят
    последним.
    """
    registry = _WORKER["registry"]
    raw_dir: Path = _WORKER["raw_dir"]

    monitor = DataQualityMonitor()
    quarantine = Quarantine(monitor, processing_time=processing_time, pipeline_version=version)
    reader = SourceReader(registry, monitor=monitor, quarantine=quarantine)
    resolver = IdentityResolver(
        registry, _WORKER["mapping"], monitor=monitor, quarantine=quarantine
    )

    partition = Partition(
        source=source, path=raw_dir / canonical_path, canonical_path=canonical_path
    )
    records = list(resolver.resolve(reader.read(partition)))

    return {
        "pid": os.getpid(),
        "partition": canonical_path,
        "records": records,
        "monitor": monitor,
        "quarantine": quarantine,
    }


# --------------------------------------------------------------------------- #
# Родитель
# --------------------------------------------------------------------------- #


def read_identified_parallel(
    raw_dir: Path,
    *,
    config_dir: Path,
    partitions: Sequence[Partition],
    workers: int,
    monitor: DataQualityMonitor,
    quarantine: Quarantine,
    processing_time: datetime,
    pipeline_version: str,
) -> tuple[list[IdentifiedRecord], ParallelEvidence]:
    """Прочитать партиции в нескольких процессах и слить их в каноническом порядке.

    Задачи ставятся по одной партиции, а не заранее нарезанными группами:
    распределение решает пул, и оно становится настоящим следом прогона, а не
    нашим предположением о нём.
    """
    if workers < 2:
        raise ParallelismNotProvenError(
            f"запрошен {workers} воркер — сравнивать multi-worker не с чем"
        )

    canonical = tuple(item.canonical_path for item in partitions)
    by_pid: dict[int, list[str]] = {}
    completion: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(config_dir), str(raw_dir)),
    ) as pool:
        futures = {
            pool.submit(
                _read_partition,
                item.canonical_path,
                item.source,
                processing_time,
                pipeline_version,
            ): item.canonical_path
            for item in partitions
        }
        for future in as_completed(futures):
            payload = future.result()
            completion.append(payload["partition"])
            by_pid.setdefault(payload["pid"], []).append(payload["partition"])
            payloads[payload["partition"]] = payload

    evidence = ParallelEvidence(
        parent_pid=os.getpid(),
        requested_workers=workers,
        by_pid={pid: tuple(sorted(items)) for pid, items in by_pid.items()},
        completion_order=tuple(completion),
        canonical_order=canonical,
    )

    records = merge_in_canonical_order(
        payloads, canonical, monitor=monitor, quarantine=quarantine
    )
    return records, evidence


def merge_in_canonical_order(
    payloads: Mapping[str, Mapping[str, Any]],
    canonical: Sequence[str],
    *,
    monitor: DataQualityMonitor,
    quarantine: Quarantine,
) -> list[IdentifiedRecord]:
    """Слить результаты воркеров строго по каноническому порядку (§29 пп. 1–3).

    Вынесено отдельной функцией не для красоты: порядок слияния — гарантия,
    которую **нельзя проверить поведением**. Всё, что ниже по цепочке,
    пересортировывает само, поэтому слияние в любом другом порядке даёт тот
    же выход, и ни один поведенческий тест не покраснеет. Проверять такую
    гарантию можно только напрямую — см.
    `test_merge_yields_canonical_order_whatever_the_workers_did`.
    """
    records: list[IdentifiedRecord] = []
    for canonical_path in canonical:
        payload = payloads[canonical_path]
        records.extend(payload["records"])
        monitor.merge(payload["monitor"])
        quarantine.merge(payload["quarantine"])
    return records
