"""Слияние результатов воркеров — §29 п.3.

Почему это критичная проверка, а не тест на геттер: §29 п.10 требует, чтобы
single-worker и multi-worker давали байт-в-байт одинаковый результат, и
проверка 4.1 на этом стоит. Счётчики и карантин собираются из воркеров в
недетерминированном порядке — если слияние окажется зависимым от порядка,
4.1 покажет «не совпало» без указания места, и искать придётся во всём
пайплайне.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor, Metric, Total
from src.preprocessing.core.quarantine import Quarantine, ReasonCode

UTC = timezone.utc
RUN_AT = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)


def make_worker(offset: int) -> DataQualityMonitor:
    """Монитор одного воркера: глобальные счётчики, знаменатели и разрез по полю."""
    monitor = DataQualityMonitor()
    monitor.count(Metric.CUTOFF_VIOLATION_RATE, by=offset)
    monitor.add_total(Total.EVENTS_VALIDATED, by=offset * 10)
    monitor.count(Metric.MISSING_RATE, label=f"field_{offset}", by=offset)
    monitor.count(Metric.MISSING_RATE, label="currency", by=offset)
    monitor.add_label_total(Metric.MISSING_RATE, "currency", by=offset * 100)
    return monitor


def merged(order: list[DataQualityMonitor]) -> dict:
    target = DataQualityMonitor()
    for worker in order:
        target.merge(worker)
    return target.report()


def test_monitor_merge_is_commutative():
    """Порядок слияния воркеров не меняет отчёт.

    Это и есть свойство, которое делает многопроцессный прогон сравнимым
    с однопроцессным.
    """
    first, second, third = make_worker(1), make_worker(2), make_worker(3)

    assert merged([first, second, third]) == merged([third, first, second])
    assert merged([first, second, third]) == merged([second, third, first])


def test_monitor_merge_sums_counters_and_totals():
    """Слияние складывает, а не перезаписывает: потерянный воркер исказил бы долю."""
    report = merged([make_worker(1), make_worker(2)])
    cutoff = report["metrics"]["cutoff_violation_rate"]

    assert cutoff["count"] == 3
    assert cutoff["denominator_value"] == 30
    assert cutoff["rate"] == 3 / 30


def test_monitor_merge_keeps_label_dimension():
    """Разрез по полю (§33.6) переживает слияние вместе со своим знаменателем."""
    by_label = merged([make_worker(1), make_worker(2)])["metrics"]["missing_rate"]["by_label"]

    assert set(by_label) == {"currency", "field_1", "field_2"}
    assert by_label["currency"]["count"] == 3
    assert by_label["currency"]["denominator_value"] == 300


def test_rate_without_denominator_is_none_not_zero():
    """«Ноль ошибок» и «ни одной попытки» — разные состояния.

    Вернув `0.0` вместо `None`, отчёт сказал бы, что компонент отработал
    безупречно, тогда как он не отработал вовсе. По дашборду это неотличимо
    от нормы — и потому опаснее ошибки.
    """
    report = DataQualityMonitor().report()

    assert report["metrics"]["numeric_parse_error_rate"]["denominator_value"] == 0
    assert report["metrics"]["numeric_parse_error_rate"]["rate"] is None

    monitor = DataQualityMonitor()
    monitor.add_total(Total.NUMERIC_VALUES, by=100)
    assert monitor.report()["metrics"]["numeric_parse_error_rate"]["rate"] == 0.0


def test_quarantine_always_raises_its_metric():
    """§34: отбросить запись и не поднять метрику технически невозможно.

    Это главное свойство карантина, а не деталь: «запрещено молча удалять
    проблемные записи» выполняется тем, что метрику инкрементит сам
    `Quarantine`, а не вызывающий код, который может забыть.
    """
    monitor = DataQualityMonitor()
    quarantine = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")

    quarantine.add(
        ReasonCode.UNKNOWN_EVENT_TYPE,
        source="core_payments",
        raw_reference="CP-1",
        detail="код вне маппинга",
    )
    report = monitor.report()

    assert report["metrics"]["unknown_event_type_rate"]["count"] == 1
    assert report["totals"]["quarantined"] == 1


def test_quarantine_metric_can_only_be_declined_explicitly():
    """Отказ от метрики возможен, но только явным `count_metric=False`.

    Так считаются метрики уровня группы (§9.3): группу уже посчитал
    вызывающий код, и запись в карантин не должна поднимать счётчик ещё раз.
    Общий счётчик карантина при этом растёт всегда — запись не теряется.
    """
    monitor = DataQualityMonitor()
    quarantine = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")

    quarantine.add(
        ReasonCode.UNRESOLVED_DUPLICATE_CONFLICT,
        source="core_payments",
        raw_reference="CP-1",
        count_metric=False,
    )
    report = monitor.report()

    assert report["metrics"]["dedup_conflict_rate"]["count"] == 0
    assert report["totals"]["quarantined"] == 1


def test_quarantine_rejects_naive_processing_time():
    """Время прогона обязано быть с зоной.

    Наивное значение означало бы неизвестную зону в файлах карантина, а они
    сравниваются побайтово при проверке single vs multi-worker (§29 п.10).
    """
    monitor = DataQualityMonitor()

    with pytest.raises(ValueError, match="часовым поясом"):
        Quarantine(monitor, processing_time=datetime(2026, 7, 28, 10, 0), pipeline_version="0.1.0")


def make_quarantine(monitor: DataQualityMonitor, reference: str) -> Quarantine:
    quarantine = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")
    quarantine.add(
        ReasonCode.UNKNOWN_EVENT_TYPE,
        source="core_payments",
        raw_reference=reference,
        detail="код вне маппинга",
    )
    return quarantine


def test_quarantine_files_do_not_depend_on_merge_order(tmp_path):
    """Файлы карантина одинаковы независимо от того, чей воркер пришёл первым.

    Порядок строк задаётся сортировкой при записи, а не порядком поступления,
    иначе побайтовое сравнение прогонов ловило бы ложные расхождения.
    """
    monitor = DataQualityMonitor()
    left = make_quarantine(monitor, "CP-000001-00001")
    right = make_quarantine(monitor, "CP-000002-00002")

    forward = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")
    forward.merge(left)
    forward.merge(right)

    backward = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")
    backward.merge(right)
    backward.merge(left)

    forward.write(tmp_path / "forward")
    backward.write(tmp_path / "backward")

    assert (tmp_path / "forward" / "core_payments.jsonl").read_bytes() == (
        tmp_path / "backward" / "core_payments.jsonl"
    ).read_bytes()
