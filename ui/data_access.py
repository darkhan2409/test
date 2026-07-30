"""Чтение того, что оставил прогон. Только чтение.

Ни одного правила препроцессинга здесь нет: модуль открывает файлы, отбирает
строки по клиенту и номеру записи и сравнивает словари. Всё, что показывает
экран, лежит в данных — реконструкции нет.

Источники: `data/debug/<NN_component>/{in,out}.jsonl`,
`data/prepared/<набор>/prepared_events.jsonl`, `data/prepared/run_report.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = ROOT / "data" / "debug"
PREPARED_DIR = ROOT / "data" / "prepared"
META_DIR = ROOT / "data" / "raw" / "_meta"

DATASET = "main"

# Шаг 1 привязывает строку входа к месту в файле, а не к номеру записи: до
# проверки первичного ключа доверять ему нельзя (так в `source_reader`).
# Поэтому вход первого шага ищется по ключу «партиция#строка», взятому из его
# же выхода. Это чтение данных, а не догадка о них.
READER_SLUG = "01_source_reader"


class DataMissingError(RuntimeError):
    """Данных прогона нет — экран не может ничего показать."""


@dataclass(frozen=True)
class EventRef:
    """Событие в списке выбора."""

    event_id: str
    event_type: str
    timestamp_utc: str
    source_record_id: str
    client_id: str
    currency: str = ""
    """Валюта операции, если у события такое поле есть. Нужна только для выбора
    события по умолчанию: на операции в валюте видно больше шагов."""

    def label(self) -> str:
        moment = self.timestamp_utc.replace("T", " ").replace("Z", "")
        tail = f" · {self.currency}" if self.currency and self.currency != "KZT" else ""
        return f"{moment} · {self.event_type}{tail} · {self.source_record_id}"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_ready() -> list[str]:
    """Чего не хватает для показа. Пустой список — всё на месте."""
    missing: list[str] = []
    if not DEBUG_DIR.is_dir() or not any(DEBUG_DIR.iterdir()):
        missing.append("data/debug/ — прогон с трассировкой не выполнялся")
    if not (PREPARED_DIR / DATASET / "prepared_events.jsonl").exists():
        missing.append(f"data/prepared/{DATASET}/prepared_events.jsonl")
    return missing


def prepared_events() -> list[dict[str, Any]]:
    path = PREPARED_DIR / DATASET / "prepared_events.jsonl"
    if not path.exists():
        raise DataMissingError(f"нет файла {path}")
    return list(_read_jsonl(path))


def traced_clients() -> set[str]:
    """Клиенты, которые есть в трассировке.

    Дамп пишется не по всем: фильтр §1.6 оставляет ролевых клиентов и обещанных
    «чистых». Показывать в выборе кого-то ещё нельзя — экран оказался бы пустым.
    Список берётся из самого дампа, а не из фильтра: фильтр — это код, а дамп —
    факт того прогона, который показываем.
    """
    for component_dir in sorted(DEBUG_DIR.iterdir()) if DEBUG_DIR.is_dir() else []:
        path = component_dir / "out.jsonl"
        if not path.exists():
            continue
        found = {row.get("client_id") for row in _read_jsonl(path)}
        found.discard(None)
        if found:
            return found  # type: ignore[return-value]
    return set()


def role_clients() -> set[str]:
    """Клиенты, за которыми закреплены краевые случаи (§29.2)."""
    path = META_DIR / "edge_case_manifest.json"
    if not path.exists():
        return set()
    cases = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["client_id"]
        for entries in cases.values()
        for entry in entries
        if entry.get("client_id") and entry["client_id"] != "-"
    }


def clean_clients() -> list[str]:
    """Клиенты, у которых нет ни одного краевого случая.

    Список объявляет сам набор: `edge_case_manifest.json` перечисляет ролевых
    клиентов, `manifest.json` — сколько чистых обещано. Тот же способ, которым
    выбирает клиентов сам debug-режим.
    """
    cases_path = META_DIR / "edge_case_manifest.json"
    identity_path = META_DIR / "identity_mapping.json"
    if not cases_path.exists() or not identity_path.exists():
        return []

    roles = role_clients()
    traced = traced_clients()
    mapping = json.loads(identity_path.read_text(encoding="utf-8"))
    everyone = sorted({value for section in mapping.values() for value in section.values()})
    # Только те, кто есть в трассировке: остальные «чистые» в дамп не попали,
    # и показывать их в выборе значило бы обещать путь, которого нет.
    return [item for item in everyone if item not in roles and item in traced]


def events_by_client(events: list[dict[str, Any]]) -> dict[str, list[EventRef]]:
    """События, разложенные по клиентам, в порядке их timeline."""
    grouped: dict[str, list[EventRef]] = {}
    for item in events:
        ref = EventRef(
            event_id=item["event_id"],
            event_type=item["event_type"],
            timestamp_utc=item["timestamp_utc"],
            source_record_id=item["ordering_key"].rsplit("|", 1)[-1],
            client_id=item["client_id"],
            currency=str(item.get("fields", {}).get("currency", "")),
        )
        grouped.setdefault(ref.client_id, []).append(ref)
    return grouped


def trace(source_record_id: str, event_id: str) -> dict[str, dict[str, Any | None]]:
    """Собрать вход и выход каждого шага для одной записи.

    Возвращает `{каталог шага: {"in": строка | None, "out": строка | None}}`.
    Отсутствие строки — тоже факт: запись через этот шаг не проходила.
    """
    if not DEBUG_DIR.is_dir():
        raise DataMissingError(f"нет каталога {DEBUG_DIR}")

    keys = {source_record_id, event_id}
    # Вход первого шага: ключ берётся из его собственного выхода.
    reader_out = _find_row(DEBUG_DIR / READER_SLUG / "out.jsonl", keys)
    if reader_out is not None:
        payload = reader_out.get("payload", {})
        partition, line = payload.get("partition"), payload.get("line_number")
        if partition and line:
            keys.add(f"{partition}#{line}")

    found: dict[str, dict[str, Any | None]] = {}
    for component_dir in sorted(DEBUG_DIR.iterdir()):
        if not component_dir.is_dir():
            continue
        found[component_dir.name] = {
            stage: _find_row(component_dir / f"{stage}.jsonl", keys)
            for stage in ("in", "out")
        }
    return found


def _find_row(path: Path, keys: set[str]) -> dict[str, Any] | None:
    """Первая строка дампа, привязанная к одному из ключей.

    Сначала грубая проверка подстрокой — она отсекает 99% строк без разбора
    JSON, и полный проход по дампу занимает меньше секунды.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not any(key in line for key in keys):
                continue
            row = json.loads(line)
            if row.get("source_record_id") in keys or row.get("event_id") in keys:
                return row
    return None


# --------------------------------------------------------------------------- #
# Сравнение вход/выход
# --------------------------------------------------------------------------- #

ABSENT = object()
"""Метка «ключа не было». Отличается от `None`: `None` — это значение."""


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Разложить вложенные словари в плоские пути `a.b.c`.

    Списки остаются значениями целиком: обрезка списка — это одно изменение
    поля, а не пять изменений его элементов.
    """
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key in value:
            flat.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


@dataclass(frozen=True)
class FieldDiff:
    path: str
    before: Any
    after: Any

    @property
    def kind(self) -> str:
        if self.before is ABSENT:
            return "added"
        if self.after is ABSENT:
            return "removed"
        return "changed"


TRACKED_IDENTITY = ("client_id", "event_id")
"""Ключи привязки, которые показываются в разнице наравне с полями.

Дамп держит их на верхнем уровне строки, а не в `payload`, поэтому без этого
появление `client_id` на шаге 2 и `event_id` на шаге 6 в разнице не видно — а
это ровно то, что происходит на этих шагах. `source_record_id` и `client_ref`
сюда не берутся: они не меняются по дороге, а на первом шаге вход привязан к
месту строки в файле, и разница показала бы техническую замену ключа.
"""


def _comparable(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    identity = {key: row[key] for key in TRACKED_IDENTITY if row.get(key) is not None}
    return flatten({**identity, **row.get("payload", {})})


def compare(before: dict[str, Any] | None, after: dict[str, Any] | None) -> tuple[list[FieldDiff], dict[str, Any]]:
    """Разница между входом и выходом шага плюс неизменившиеся поля."""
    left = _comparable(before)
    right = _comparable(after)

    diffs: list[FieldDiff] = []
    same: dict[str, Any] = {}
    for path in sorted(set(left) | set(right)):
        old = left.get(path, ABSENT)
        new = right.get(path, ABSENT)
        if old == new:
            same[path] = new
        else:
            diffs.append(FieldDiff(path=path, before=old, after=new))
    return diffs, same


def run_report() -> dict[str, Any] | None:
    path = PREPARED_DIR / "run_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepared_event(event_id: str) -> dict[str, Any] | None:
    for item in prepared_events():
        if item["event_id"] == event_id:
            return item
    return None
