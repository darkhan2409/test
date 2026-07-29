"""CutoffFilter — §14, §14.1, §14.2.

Одно правило: в обработку идут только записи с `timestamp_utc <= T`.
Граница включающая — событие ровно в T входит.

Почему отсечка стоит так рано в цепочке (§37.2), а не перед выдачей: §14
перечисляет, что обязано считаться уже по отсечённым данным — выбор профиля,
агрегаты, life-long признаки, временные дельты, сессии, бакетизация. Если
отсечь в конце, каждый из этих расчётов успеет увидеть будущее, и leakage
попадёт в признаки, оставшись невидимым в самих событиях.

Фильтр применяется ко **всем** источникам, включая курсы валют: §18 запрещает
брать курс позже события, а все события уже не позже T. Курс за 5 февраля не
нужен ни одному событию до 31 января, и держать его — значит держать
возможность ошибиться.

Отброшенная запись — не брак. Она не идёт в карантин и не поднимает метрику
ошибки: находиться после T нормально, это просто вне окна наблюдения.
Считается она отдельным отчётом, чтобы «пропало 300 записей» не выглядело
загадкой.

`cutoff_violation_rate` (§33.1) — не счётчик отброшенного, а сторож: он
считает записи, которые оказались после T **уже после** фильтрации. По
построению таких нет; §33.1 и говорит «любое значение > 0 — critical».
Ту же метрику поднимают Sessionizer (§14.1) и ProfileBuilder (§6 п.1) для
своих случаев утечки.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Metric, Total
from .records import TimedRecord

UTC = timezone.utc

COMPONENT = "cutoff_filter"

CUTOFF_POLICY_VERSION = "1.0.0"


def cutoff_policy_state(cutoff: datetime) -> dict[str, Any]:
    """Cutoff policy — пункт перечней §30 и §31.

    Значение `T` здесь то же, что в `run_policy`: оба вида берутся из одного
    экземпляра настроек в один момент, разойтись им негде. Дублирование
    осознанное — §31 требует хранить cutoff policy как самостоятельный
    артефакт, а политика без самой границы ничего не описывает.
    """
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff должен быть с часовым поясом")
    return {
        "cutoff_policy_version": CUTOFF_POLICY_VERSION,
        "cutoff_time_utc": cutoff.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        # §14: событие ровно в T входит, T + 1 секунда — нет.
        "boundary": "inclusive",
        # Единый глобальный T на прогон, а не per-client окно.
        "scope": "global_single_t",
        # Решение шага 2.4: фильтр применяется и к справочникам тоже, иначе
        # §18 мог бы взять курс позже события.
        "applies_to": "all_sources",
    }


class CutoffLeakageError(RuntimeError):
    """Запись после T дошла до выхода фильтра.

    Возможна только при ошибке в коде, но проверять дешевле, чем ловить
    последствия: утечка будущего в признаки не видна ни в одном отчёте.
    """


@dataclass
class CutoffReport:
    """Сколько записей осталось за границей окна наблюдения."""

    cutoff_time: datetime
    kept: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "CutoffReport") -> None:
        self.kept.update(other.kept)
        self.dropped.update(other.dropped)

    def summary(self) -> dict[str, Any]:
        return {
            "cutoff_time": self.cutoff_time.isoformat().replace("+00:00", "Z"),
            "kept": dict(sorted(self.kept.items())),
            "dropped_after_cutoff": dict(sorted(self.dropped.items())),
            "dropped_total": sum(self.dropped.values()),
        }


class CutoffFilter:
    """Отсечка по времени T."""

    def __init__(
        self,
        *,
        cutoff: datetime,
        monitor: DataQualityMonitor,
        debug: DebugDump | None = None,
    ) -> None:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff должен быть с часовым поясом")
        self.cutoff = cutoff
        self._monitor = monitor
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self.report = CutoffReport(cutoff_time=cutoff)

    def apply(self, records: Iterable[TimedRecord]) -> Iterator[TimedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            if record.timestamp_utc is None:
                raise CutoffLeakageError(
                    f"{record.raw_reference}: запись без timestamp_utc дошла до отсечки — "
                    "§12 обязан был отправить её в карантин"
                )

            self._monitor.add_total(Total.EVENTS_VALIDATED)

            if record.timestamp_utc > self.cutoff:
                self.report.dropped[record.source] += 1
                continue

            self.report.kept[record.source] += 1
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [record.debug_row()])
            yield record

    def verify(self, records: Iterable[TimedRecord]) -> Iterator[TimedRecord]:
        """Сторож §33.1 на готовом наборе.

        Отдельный проход, потому что утечку может внести не только фильтр:
        §14.1 (сессия за T) и §6 п.1 (будущий снимок профиля) добавляют
        значения после отсечки, не трогая сами события.
        """
        for record in records:
            if record.timestamp_utc is not None and record.timestamp_utc > self.cutoff:
                self._monitor.count(Metric.CUTOFF_VIOLATION_RATE)
                raise CutoffLeakageError(
                    f"{record.raw_reference}: {record.timestamp_utc.isoformat()} позже T "
                    f"{self.cutoff.isoformat()} (§14, §33.1)"
                )
            yield record
