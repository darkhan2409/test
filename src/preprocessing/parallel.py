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

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core.monitor import DataQualityMonitor
from .core.quarantine import Quarantine
from .identity_resolver import IdentityResolver, IdentityTable, load_identity_mapping
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

    start_timed_out: bool = False
    """Хотя бы один воркер не дождался остальных на барьере старта.

    Отдельный след, а не вывод из распределения. Истёкший барьер означает, что
    пул собрался не полностью, и работа могла разойтись как угодно — в том
    числе прилично выглядящим образом. Считать такой прогон состоявшимся
    нельзя, а по одному только `by_pid` этого не видно."""

    def summary(self) -> dict[str, Any]:
        return {
            "parent_pid": self.parent_pid,
            "requested_workers": self.requested_workers,
            "actual_processes": len(self.by_pid),
            "start_timed_out": self.start_timed_out,
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

        if self.start_timed_out:
            problems.append(
                f"воркеры не собрались за {WORKER_START_TIMEOUT:.0f} c — пул поднялся "
                "не полностью, и раскладка партиций ничего не говорит о параллельности"
            )

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


def require_parallelism(evidence: ParallelEvidence, *, attempts: int = 1) -> None:
    """Убедиться, что сравнивать вообще было что.

    Ничего не возвращает: единственный исход, кроме молчаливого успеха, —
    исключение. Вердикта «параллельность слабая, но засчитаем» не существует.
    """
    problems = evidence.degeneracies()
    if problems:
        tried = "" if attempts == 1 else f" за {attempts} попыт(ок)"
        raise ParallelismNotProvenError(
            f"проверка §29 п.10 не проведена, а не пройдена{tried} — "
            + "; ".join(problems)
            + f". Следы прогона: {evidence.summary()}"
        )


PARALLELISM_ATTEMPTS = 3
"""Сколько раз повторить параллельный прогон, пока опыт не состоится."""


def run_until_conducted(
    attempt: Callable[[], tuple[Any, ParallelEvidence]], *, attempts: int = PARALLELISM_ATTEMPTS
) -> tuple[Any, ParallelEvidence]:
    """Повторять параллельный прогон, пока проверка §29 п.10 не состоится.

    «Не проведена» — это отсутствие условий для опыта, а не его исход.
    Раскладку партиций по процессам решает планировщик ОС, и вырожденная
    раскладка означает, что опыт не поставлен; поставить его заново —
    нормальная реакция, а не сокрытие результата. Повторяется **только**
    параллельная половина: однопроцессный прогон детерминирован, повторять
    его незачем.

    Повторов конечное число. Если условия не складываются раз за разом, это
    уже свойство машины, и объявлять проверку пройденной нельзя — поэтому
    последняя попытка отдаётся `require_parallelism`, а та поднимает
    `ParallelismNotProvenError`. Он отличается от `ParallelOutputMismatchError`
    типом, и разница обязана доживать до вердикта прогона: «опыт не удалось
    поставить» и «выходы разошлись» — разные новости.
    """
    for _ in range(attempts):
        result, evidence = attempt()
        if not evidence.degeneracies():
            return result, evidence

    require_parallelism(evidence, attempts=attempts)
    raise AssertionError("require_parallelism обязан был подняться")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Воркер
# --------------------------------------------------------------------------- #

_WORKER: dict[str, Any] = {}


WORKER_START_TIMEOUT = 30.0
"""Предел ожидания на барьере старта.

Барьер без предела виснет насмерть, если воркер не поднялся вовсе, — а
безнадёжное ожидание в этом проекте уже дважды съедало десятки минут. Поэтому
предел есть, и превышение — не зависание и не отказ, а **непроведённая
проверка**: барьер ломается для всех сразу (`BrokenBarrierError` приходит
каждому ждущему немедленно, а не по своему таймауту), прогон идёт дальше, факт
уезжает в следы полем `start_timed_out`, и `degeneracies()` называет его
вслух.

Тридцать секунд — запас в десятки раз: измеренный прогон пула на 148 партиций
с четырьмя воркерами держится в полсекунды даже под нагрузкой. Срабатывает
предел только когда что-то сломано, и тогда важно, чтобы он всё же
сработал."""


def _init_worker(
    config_dir: str,
    raw_dir: str,
    identity_table: IdentityTable,
    started: Any = None,
    start_timeout: float = WORKER_START_TIMEOUT,
) -> None:
    """Загрузить контракты один раз на процесс.

    Загрузка внутри воркера, а не передача объектов из родителя, — это ещё и
    проверка: разбор конфигов обязан быть детерминированным, иначе четыре
    процесса получили бы четыре немного разных контракта.

    Источник таблицы identity приезжает из родителя, а не выводится здесь:
    вывести его воркер может только из своего текущего каталога, а он у
    процесса пула не обязан совпадать ни с чем осмысленным.

    По той же причине из родителя приезжает и `start_timeout`. Воркер — это
    `spawn`-процесс: он импортирует модуль заново и видит константу по
    умолчанию, а не то, что настроил родитель. Пока предел жил только в
    модуле, задать его снаружи было нельзя, и проверка сквозного поведения
    измеряла не тот предел, который объявляла.
    """
    registry = load_source_contracts(Path(config_dir) / "source_contracts.yaml")
    mapping = load_identity_mapping(
        Path(config_dir) / "identity_mapping.yaml", registry, table=identity_table
    )
    _WORKER["registry"] = registry
    _WORKER["mapping"] = mapping
    _WORKER["raw_dir"] = Path(raw_dir)

    _WORKER["start_timed_out"] = False
    if started is not None:
        # Барьер: пока не поднялись все воркеры, работу не берёт никто.
        # Иначе распределение решает не пул, а гонка — см. `_start_barrier`.
        try:
            started.wait(timeout=start_timeout)
        except threading.BrokenBarrierError:
            # Кто-то не дошёл. Падать здесь нельзя — упадёт весь пул, и вместо
            # «проверка не проведена» получится «прогон сломался». Факт
            # запоминается и уезжает наверх с каждой партицией: решение о том,
            # засчитывать ли прогон, принимает `degeneracies()`, а не воркер.
            _WORKER["start_timed_out"] = True


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
        "start_timed_out": _WORKER.get("start_timed_out", False),
    }


# --------------------------------------------------------------------------- #
# Родитель
# --------------------------------------------------------------------------- #


def _start_barrier(context: Any, workers: int, partitions: int) -> Any:
    """Барьер, на котором воркеры дожидаются друг друга перед первой задачей.

    Без него распределение партиций решает не пул, а гонка со стартом
    процессов. `ProcessPoolExecutor` поднимает воркеров лениво, старт процесса
    под Windows (`spawn` плюс импорт пакета) стоит на порядки дороже чтения
    одной партиции, и первый поднявшийся успевает разобрать очередь до
    появления остальных. Измерено: на незагруженной машине распределение
    выходило 4/4 из 4 воркеров, под нагрузкой конформанса — 2 процесса из 4 и
    порядок завершения, совпавший с каноническим. Проверка §29 п.10 при этом
    объявляла себя непроведённой — по причине, к предмету проверки отношения
    не имеющей.

    Барьер передаётся через `initargs`, а не через `Manager`: аргументы
    инициализатора уезжают в воркер как аргументы `Process`, то есть
    наследованием, — единственный способ, которым примитивы синхронизации
    вообще разрешено передавать.

    `None`, когда партиций меньше, чем воркеров: часть процессов не получит ни
    одной задачи и до барьера не дойдёт, а ждать их — значит гарантированно
    выстоять таймаут. Распределение в этом случае вырождено по построению, и
    называет это `require_parallelism`, а не барьер.
    """
    if partitions < workers:
        return None
    return context.Barrier(workers)


def read_identified_parallel(
    raw_dir: Path,
    *,
    config_dir: Path,
    identity_table: IdentityTable,
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
    start_timed_out = False

    context = multiprocessing.get_context("spawn")
    started = _start_barrier(context, workers, len(partitions))

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            str(config_dir), str(raw_dir), identity_table, started, WORKER_START_TIMEOUT
        ),
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
            start_timed_out = start_timed_out or payload.get("start_timed_out", False)

    evidence = ParallelEvidence(
        parent_pid=os.getpid(),
        requested_workers=workers,
        by_pid={pid: tuple(sorted(items)) for pid, items in by_pid.items()},
        completion_order=tuple(completion),
        canonical_order=canonical,
        start_timed_out=start_timed_out,
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
