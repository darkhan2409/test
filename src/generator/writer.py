"""Запись сырых данных в JSONL-партиции.

Партиция — `data/raw/<source>/<YYYY-MM-DD>.jsonl`. Такая раскладка нужна не для
красоты: §29 п.1 требует, чтобы препроцессинг обходил файлы в стабильном порядке
по каноническому пути, а многопроцессная обработка резала работу по партициям.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from .records import RawRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileInfo:
    """Строка манифеста: что записали и с какой контрольной суммой."""

    path: str
    records: int
    sha256: str


class JsonlWriter:
    """Пишет RawRecord в партиции детерминированно.

    Три вещи, от которых зависит побайтовая одинаковость файлов:
    порядок записей внутри партиции (сортировка по `sort_key`),
    перевод строки (всегда `\\n`, а не CRLF от Windows) и
    порядок ключей в JSON (`sort_keys=True`).
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def write(self, records: Iterable[RawRecord]) -> list[FileInfo]:
        partitions = self._group(records)
        self._clear_previous_run({source for source, _ in partitions})

        written: list[FileInfo] = []
        for (source, partition_date) in sorted(partitions):
            rows = sorted(partitions[(source, partition_date)], key=lambda item: item.sort_key)
            written.append(self._write_partition(source, partition_date, rows))
        return written

    @staticmethod
    def _group(records: Iterable[RawRecord]) -> dict[tuple[str, date], list[RawRecord]]:
        grouped: dict[tuple[str, date], list[RawRecord]] = defaultdict(list)
        for record in records:
            grouped[(record.source, record.partition_date)].append(record)
        return grouped

    def _clear_previous_run(self, sources: set[str]) -> None:
        """Удалить JSONL прошлого прогона.

        Без этого файлы старой конфигурации (например, с другим периодом
        истории) остались бы рядом с новыми, и датасет перестал бы
        соответствовать манифесту.
        """
        for source in sorted(sources):
            source_dir = self.out_dir / source
            if not source_dir.exists():
                continue
            stale = sorted(source_dir.glob("*.jsonl"))
            for path in stale:
                path.unlink()
            if stale:
                logger.info("%s: удалено %d файлов прошлого прогона", source, len(stale))

    def _write_partition(self, source: str, partition_date: date, rows: list[RawRecord]) -> FileInfo:
        source_dir = self.out_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{partition_date.isoformat()}.jsonl"

        payload = "".join(
            json.dumps(row.payload, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        raw = payload.encode("utf-8")
        path.write_bytes(raw)

        return FileInfo(
            path=f"{source}/{path.name}",
            records=len(rows),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
