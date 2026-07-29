"""Общий контейнер сырой записи, который источники отдают писателю."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class RawRecord:
    """Одна строка сырого источника до всякой обработки.

    `payload` намеренно нетипизирован: у каждого источника своя схема, и
    приводить её к общему виду — работа препроцессинга (§4, §5), а не генератора.
    """

    source: str
    partition_date: date
    sort_key: str
    payload: dict[str, Any]
