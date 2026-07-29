"""Доказательство параллельности — §29 п.10, §27 шаг 16.

Сравнение выходов 1 и 4 воркеров доказывает детерминизм **только** если
работа реально разделилась. Если все партиции достались одному процессу или
пул отработал их в каноническом порядке, сравнение сравнивает прогон с самим
собой и всегда зелёное — шестая по счёту маска вырожденной проверки в этом
проекте.

Поэтому вырожденность объявляется не «не пройдено», а **«не проведено»**, и у
неё свой тип ошибки: `ParallelismNotProvenError` нельзя спутать с
`ParallelOutputMismatchError`, потому что это разные факты.

Сам параллельный прогон живёт в проверке шага 4.1: ему нужны данные и четыре
процесса. Здесь проверяется разбор следов — то, что решает, засчитывать
прогон или нет.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.core.quarantine import Quarantine
from src.preprocessing.parallel import (
    ParallelEvidence,
    ParallelismNotProvenError,
    merge_in_canonical_order,
    require_parallelism,
)

CANONICAL = ("a/1.jsonl", "a/2.jsonl", "b/1.jsonl", "b/2.jsonl")
SHUFFLED = ("b/1.jsonl", "a/1.jsonl", "b/2.jsonl", "a/2.jsonl")


def evidence(**overrides) -> ParallelEvidence:
    values = {
        "parent_pid": 1000,
        "requested_workers": 4,
        "by_pid": {2001: CANONICAL[:2], 2002: CANONICAL[2:]},
        "completion_order": SHUFFLED,
        "canonical_order": CANONICAL,
    }
    values.update(overrides)
    return ParallelEvidence(**values)


def test_healthy_run_is_accepted():
    """Разные процессы и переставленный порядок завершения — проверка проведена."""
    assert require_parallelism(evidence()) is None


def test_single_process_means_the_check_was_not_conducted():
    """Все партиции одному процессу — сравнивать было нечего."""
    with pytest.raises(ParallelismNotProvenError, match="не разделилась"):
        require_parallelism(evidence(by_pid={2001: CANONICAL}))


def test_work_in_the_parent_process_means_the_pool_never_started():
    """Родительский PID среди исполнителей — пул не запустился.

    Отдельная причина, а не частный случай предыдущей: процессов может быть
    и два, но если один из них родительский, часть работы шла последовательно.
    """
    with pytest.raises(ParallelismNotProvenError, match="родительским процессом"):
        require_parallelism(
            evidence(by_pid={1000: CANONICAL[:2], 2002: CANONICAL[2:]})
        )


def test_canonical_completion_order_means_reordering_was_not_exercised():
    """Порядок завершения совпал с каноническим — переупорядочивание не проверено.

    Выход при этом совпадёт обязательно: слияние идёт по каноническому
    порядку, и если задачи завершились в нём же, прогон ничем не отличался от
    однопоточного.
    """
    with pytest.raises(ParallelismNotProvenError, match="порядок завершения"):
        require_parallelism(evidence(completion_order=CANONICAL))


def test_the_message_says_not_conducted_not_failed():
    """Формулировка — часть проверки.

    «Не пройдена» читается как «детерминизм нарушен» и отправляет искать баг
    в слиянии; «не проведена» отправляет чинить прогон. Это разные действия,
    и путать их дороже, чем кажется.
    """
    with pytest.raises(ParallelismNotProvenError) as error:
        require_parallelism(evidence(by_pid={2001: CANONICAL}, completion_order=CANONICAL))

    text = str(error.value)
    assert "не проведена, а не пройдена" in text
    assert "не разделилась" in text and "порядок завершения" in text


def test_degeneracies_are_reported_together():
    """Все причины сразу, а не первая попавшаяся: чинить придётся обе."""
    problems = evidence(
        by_pid={1000: CANONICAL}, completion_order=CANONICAL
    ).degeneracies()

    assert len(problems) == 3


def test_merge_yields_canonical_order_whatever_the_workers_did():
    """Слияние отдаёт канонический порядок при любом порядке завершения.

    Сторож на **гарантию**, а не на поведение, и это осознанно. Мутация
    «слить по порядку завершения» проверена отдельно: она оставляет зелёными
    и unit-тесты, и полный прогон 4.1, потому что всё ниже по цепочке
    пересортировывает само — дедупликация выдаёт записи в каноническом
    порядке, TimelineBuilder сортирует внутри клиента, сэмплер
    порядконезависим. Значит убрать канонический порядок можно, и ни один
    поведенческий тест не покраснеет.

    Гарантию требует §29 п.3, и держится она не на сегодняшних компонентах,
    а на будущих: полагаться на то, что каждый следующий не забудет
    отсортироваться, — не то же самое, что не зависеть от порядка. Поэтому
    проверяется напрямую то, что функция обещает.
    """
    canonical = ("a/1.jsonl", "a/2.jsonl", "b/1.jsonl", "b/2.jsonl")
    # Словарь заполняется в порядке «завершения» — обратном каноническому.
    payloads = {
        path: {"records": [path], "monitor": DataQualityMonitor(), "quarantine": _quarantine()}
        for path in reversed(canonical)
    }
    assert tuple(payloads) != canonical, (
        "порядок в payloads совпал с каноническим — тест не отличит слияние "
        "по канону от слияния в порядке поступления"
    )

    merged = merge_in_canonical_order(
        payloads, canonical, monitor=DataQualityMonitor(), quarantine=_quarantine()
    )

    assert tuple(merged) == canonical


def _quarantine(monitor: DataQualityMonitor | None = None) -> Quarantine:
    from datetime import datetime, timezone

    return Quarantine(
        monitor or DataQualityMonitor(),
        processing_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        pipeline_version="0.1.0",
    )


def test_merge_keeps_every_quarantined_record_of_every_worker():
    """Карантин воркеров доезжает до родителя целиком (§34).

    Проверяется здесь, а не прогоном, потому что прогон это **не
    проверяет**: на текущих данных чтение не бракует ни одной записи, и
    `Quarantine.merge` отработал 152 раза с нулём записей. То есть путь,
    по которому брак из воркера попадает в отчёт, до сих пор не исполнялся —
    а §34 запрещает терять записи молча именно на нём.
    """
    from src.preprocessing.core.quarantine import ReasonCode

    canonical = ("a/1.jsonl", "a/2.jsonl")
    payloads = {}
    for index, path in enumerate(canonical):
        monitor = DataQualityMonitor()
        quarantine = _quarantine(monitor)
        for number in range(index + 1):
            quarantine.add(
                ReasonCode.SOURCE_CONTRACT_VIOLATION,
                source="core_payments",
                raw_reference=f"{path}#{number}",
                partition=path,
            )
        payloads[path] = {"records": [], "monitor": monitor, "quarantine": quarantine}

    parent_monitor = DataQualityMonitor()
    parent = _quarantine(parent_monitor)

    merge_in_canonical_order(
        payloads, canonical, monitor=parent_monitor, quarantine=parent
    )

    # Один брак в первой партиции, два во второй — суммарно три.
    assert parent.summary()["total"] == 3
    assert parent.summary()["by_reason"] == {"source_contract_violation": 3}
    assert parent_monitor.report()["totals"]["quarantined"] == 3


def test_summary_shows_the_actual_split():
    """Следы прогона читаемы: сколько процессов и сколько партиций каждому.

    Без этого «проверка прошла» опирается на веру, а отчёт §31 не отвечает
    на вопрос, что именно было проверено.
    """
    summary = evidence().summary()

    assert summary["actual_processes"] == 2
    assert summary["partitions_by_pid"] == {"2001": 2, "2002": 2}
    assert summary["completion_order_differs_from_canonical"] is True


def test_single_worker_request_is_refused():
    """Один воркер — сравнивать multi-worker не с чем.

    Отказ, а не тихий последовательный прогон: «сравнили и совпало» на одном
    процессе выглядело бы пройденной проверкой §29 п.10.
    """
    from src.preprocessing.parallel import read_identified_parallel

    with pytest.raises(ParallelismNotProvenError, match="сравнивать multi-worker не с чем"):
        read_identified_parallel(
            Path("data/raw"), config_dir=Path("config"), partitions=(), workers=1,
            monitor=DataQualityMonitor(), quarantine=_quarantine(),
            processing_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
            pipeline_version="0.1.0",
        )


def test_monitor_and_quarantine_are_passed_only_together():
    """Карантин обязан поднимать метрики в тот же монитор (§34).

    Передать один без другого — значит получить карантин, который считает
    брак в чужой счётчик: записи в отчёте есть, метрики нет.
    """
    from src.preprocessing.pipeline import BuildPhaseError, run_build

    with pytest.raises(BuildPhaseError, match="только вместе"):
        run_build(
            None,  # до чтения набора дело не дойдёт
            config_dir=Path("config"),
            settings=None,
            processing_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
            monitor=DataQualityMonitor(),
        )
