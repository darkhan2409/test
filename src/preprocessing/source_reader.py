"""SourceReader — §4, §29 п.1.

Первый компонент цепочки §37.2. Делает ровно три вещи и ни одной больше:

1. находит партиции в детерминированном порядке (§29 пп. 1, 2);
2. проверяет каждую запись на соответствие Source Contract;
3. отдаёт валидные записи с lineage, невалидные — в карантин (§34).

Чего он сознательно **не** делает: не разбирает время, не приводит суммы,
не нормализует валюты, не разрешает `client_id`. Это отдельные компоненты
цепочки, и попытка «заодно распарсить дату» здесь сделала бы §12 неявным.
Тип колонки проверяется, значение — нет: `"abc"` в сумме нарушает §17,
а не контракт, и должен дойти до NumericValidator живым.

Про порядок. Партиция — единица параллельной работы: воркер получает
партицию целиком. Поэтому `discover_partitions` отделён от чтения — список
партиций нужен до того, как что-то прочитано, и он обязан быть одинаковым
при любом числе воркеров. Сортировка идёт по каноническому относительному
пути в POSIX-виде, а не по тому, что вернула файловая система.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor, Total
from .core.quarantine import Quarantine, ReasonCode
from .records import SourceRecord
from .schema.source_contract import (
    ColumnType,
    SourceContract,
    SourceContractRegistry,
    UnknownFieldPolicy,
)

COMPONENT = "source_reader"
PARTITION_SUFFIX = ".jsonl"

# Разделитель составного source_record_id. Это идентификатор для lineage,
# а не пре-образ хэша, поэтому склейка допустима — но только вместе с
# проверкой, что сам разделитель в значениях ключа не встречается. Без неё
# ("a|b") и ("a", "b") дали бы один и тот же ID.
KEY_SEPARATOR = "|"

# Какому типу колонки какой Python-тип соответствует после json.loads.
# bool исключён отдельно: в Python bool — подкласс int, и `true` молча
# прошёл бы проверку на integer.
_TYPE_CHECKS: dict[ColumnType, tuple[type, ...]] = {
    ColumnType.STRING: (str,),
    ColumnType.INTEGER: (int,),
    ColumnType.STRING_LIST: (list,),
}


class SourceContractError(RuntimeError):
    """Блокирующая ошибка контракта — прогон продолжать нельзя.

    Поднимается там, где §4 требует «остановить несовместимый pipeline»:
    неизвестный источник, отсутствующий каталог, неизвестное поле при
    политике `fail`.
    """


@dataclass(frozen=True)
class Partition:
    """Единица чтения и единица параллельной работы."""

    source: str
    path: Path
    canonical_path: str
    """Относительный путь в POSIX-виде — ключ сортировки §29 п.1.
    Именно относительный: абсолютный содержал бы каталог машины, и порядок
    зависел бы от того, где лежит проект."""

    def __str__(self) -> str:
        return self.canonical_path


class SourceReader:
    """Чтение источников по контрактам."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        *,
        monitor: DataQualityMonitor,
        quarantine: Quarantine,
        debug: DebugDump | None = None,
    ) -> None:
        self.registry = registry
        self._monitor = monitor
        self._quarantine = quarantine
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))
        self._unknown_fields: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    # Обход файлов
    # ------------------------------------------------------------------ #

    def discover_partitions(self, raw_dir: Path) -> list[Partition]:
        """Все партиции в порядке §29 пп. 1, 2.

        Состав источников берётся из контрактов, а не из листинга каталога:
        каталог без контракта — данные, о которых никто не договаривался.
        """
        if not raw_dir.is_dir():
            raise SourceContractError(f"нет каталога сырых данных: {raw_dir}")

        self._check_directories(raw_dir)

        partitions: list[Partition] = []
        for source in sorted(self.registry.sources):
            contract = self.registry.contract(source)
            if contract.format != "jsonl":
                raise SourceContractError(
                    f"{source}: формат {contract.format!r} не поддерживается, ожидается jsonl"
                )

            source_dir = raw_dir / source
            if not source_dir.is_dir():
                raise SourceContractError(
                    f"{source}: контракт есть, каталога {source_dir} нет — "
                    "молча потерять источник целиком нельзя"
                )

            unexpected = sorted(
                item.name
                for item in source_dir.iterdir()
                if item.is_file() and item.suffix != PARTITION_SUFFIX
            )
            if unexpected:
                raise SourceContractError(
                    f"{source}: посторонние файлы в каталоге источника: " + ", ".join(unexpected)
                )

            nested = sorted(item.name for item in source_dir.iterdir() if item.is_dir())
            if nested:
                # Обход не рекурсивный: раскладка плоская по допущению A1,
                # `<source>/<дата>.jsonl`. Это временная гарантия — раскладка
                # может измениться, — и без этой проверки вложенные данные
                # просто не нашлись бы: `glob` вернул бы пусто, источник
                # прочитался бы «успешно», а в отчёте оказался бы честный
                # ноль записей. Пустой источник без подкаталогов при этом
                # остаётся законным: новый источник до первой загрузки.
                raise SourceContractError(
                    f"{source}: в {source_dir} есть подкаталоги ("
                    + ", ".join(nested)
                    + f"), а обход партиций плоский — файлы {PARTITION_SUFFIX} внутри них "
                    "не будут прочитаны. Смена раскладки требует осознанной правки обхода"
                )

            for path in sorted(source_dir.glob(f"*{PARTITION_SUFFIX}")):
                partitions.append(
                    Partition(
                        source=source,
                        path=path,
                        canonical_path=path.relative_to(raw_dir).as_posix(),
                    )
                )

        partitions.sort(key=lambda item: item.canonical_path)
        return partitions

    def _check_directories(self, raw_dir: Path) -> None:
        """Убедиться, что в сырых данных нет источников без контракта."""
        ignored = set(self.registry.ignored_paths)
        known = set(self.registry.sources)

        strangers = sorted(
            item.name
            for item in raw_dir.iterdir()
            if item.is_dir() and item.name not in known and item.name not in ignored
        )
        if strangers:
            raise SourceContractError(
                "в data/raw есть каталоги без Source Contract (§4): "
                + ", ".join(strangers)
                + ". Добавьте контракт или внесите путь в ignored_paths."
            )

    # ------------------------------------------------------------------ #
    # Чтение
    # ------------------------------------------------------------------ #

    def read_all(self, raw_dir: Path) -> Iterator[SourceRecord]:
        """Прочитать все партиции подряд — путь single-worker."""
        for partition in self.discover_partitions(raw_dir):
            yield from self.read(partition)

    def read(self, partition: Partition) -> Iterator[SourceRecord]:
        """Прочитать одну партицию: строки в порядке файла.

        Дамп пишется построчно и только при включённом debug: накопить его
        «на всякий случай» значило бы платить за режим, который выключен.
        """
        contract = self.registry.contract(partition.source)
        tracing = self._debug.enabled

        with partition.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue

                self._monitor.add_total(Total.RECORDS_READ)
                if tracing:
                    self._debug.record(
                        COMPONENT,
                        Stage.IN,
                        [self._debug_input(partition, line_number, contract, text)],
                    )

                record = self._validate(contract, partition, line_number, text)
                if record is None:
                    continue
                if tracing:
                    self._debug.record(COMPONENT, Stage.OUT, [record.debug_row()])
                yield record

    def _debug_input(
        self,
        partition: Partition,
        line_number: int,
        contract: SourceContract,
        text: str,
    ) -> dict[str, Any]:
        """Строка дампа «вход»: сырой JSON как есть, до всякой проверки."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"raw_line": text}
        if not isinstance(payload, dict):
            payload = {"raw_line": text}

        return {
            # До проверки первичного ключа доверять ему нельзя, поэтому
            # привязка идёт по месту строки в файле.
            "source_record_id": f"{partition.canonical_path}#{line_number}",
            "client_ref": self._client_ref(contract, payload),
            "source": contract.name,
            "partition": partition.canonical_path,
            "line_number": line_number,
            "record": payload,
        }

    # ------------------------------------------------------------------ #
    # Проверка записи
    # ------------------------------------------------------------------ #

    def _validate(
        self,
        contract: SourceContract,
        partition: Partition,
        line_number: int,
        text: str,
    ) -> SourceRecord | None:
        location = f"{partition.canonical_path}#{line_number}"

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            self._reject(
                contract, partition, location, ReasonCode.SOURCE_CONTRACT_VIOLATION,
                f"строка не разбирается как JSON: {error.msg}",
            )
            return None

        if not isinstance(payload, dict):
            self._reject(
                contract, partition, location, ReasonCode.SOURCE_CONTRACT_VIOLATION,
                f"ожидался JSON-объект, получен {type(payload).__name__}",
            )
            return None

        # Версия схемы проверяется первой: если источник уехал на другую
        # версию, все прочие расхождения — её следствие, и сообщать про них
        # по отдельности только уводит в сторону.
        schema_version = contract.schema_version
        if contract.schema_version_field is not None:
            declared = payload.get(contract.schema_version_field)
            if declared != contract.schema_version:
                self._reject(
                    contract, partition, location, ReasonCode.INCOMPATIBLE_SCHEMA_VERSION,
                    f"источник сообщает версию схемы {declared!r}, "
                    f"контракт версии {contract.schema_version!r}",
                )
                return None
            schema_version = str(declared)

        violations = self._structural_violations(contract, payload)
        if violations:
            self._reject(
                contract, partition, location, ReasonCode.SOURCE_CONTRACT_VIOLATION,
                "; ".join(violations),
            )
            return None

        return SourceRecord(
            source=contract.name,
            partition=partition.canonical_path,
            line_number=line_number,
            source_record_id=self._record_id(contract, payload),
            source_schema_version=schema_version,
            client_ref=self._client_ref(contract, payload),
            payload=payload,
        )

    def _structural_violations(
        self, contract: SourceContract, payload: Mapping[str, Any]
    ) -> list[str]:
        """Все расхождения записи со схемой, в детерминированном порядке.

        Собираются все, а не первое попавшееся: карантинная запись читается
        человеком, и «нет ключа X» без упоминания ещё трёх проблем заставляет
        чинить источник в четыре захода.
        """
        violations: list[str] = []

        unknown = sorted(set(payload) - set(contract.columns))
        if unknown:
            self._note_unknown(contract, unknown)
            if contract.unknown_field_policy is UnknownFieldPolicy.FAIL:
                raise SourceContractError(
                    f"{contract.name}: поля вне Source Contract: {', '.join(unknown)}. "
                    "Схема источника изменилась без новой версии контракта (§4)."
                )
            violations.append("поля вне контракта: " + ", ".join(unknown))

        for name in sorted(contract.columns):
            spec = contract.columns[name]
            if name not in payload:
                if spec.required:
                    violations.append(f"нет обязательного поля {name}")
                continue

            value = payload[name]
            if value is None:
                if not spec.nullable:
                    violations.append(f"{name}: null при nullable: false")
                continue

            expected = _TYPE_CHECKS[spec.type]
            if isinstance(value, bool) or not isinstance(value, expected):
                violations.append(
                    f"{name}: ожидался {spec.type}, получен {type(value).__name__}"
                )
                continue

            if spec.type is ColumnType.STRING_LIST and not all(
                isinstance(item, str) for item in value
            ):
                violations.append(f"{name}: список содержит не только строки")

        violations.extend(self._key_violations(contract, payload))
        return violations

    def _key_violations(self, contract: SourceContract, payload: Mapping[str, Any]) -> list[str]:
        """Проверить пригодность первичного ключа как ссылки на запись."""
        problems: list[str] = []
        for column in contract.primary_key:
            value = payload.get(column)
            if not isinstance(value, str) or not value.strip():
                # Тип уже проверен выше; сюда попадает пустая строка.
                if isinstance(value, str):
                    problems.append(f"primary_key {column}: пустое значение")
                continue
            if len(contract.primary_key) > 1 and KEY_SEPARATOR in value:
                problems.append(
                    f"primary_key {column}: значение содержит {KEY_SEPARATOR!r}, "
                    "составной идентификатор перестал бы быть однозначным"
                )
        return problems

    def _note_unknown(self, contract: SourceContract, unknown: Iterable[str]) -> None:
        """Накопить data-quality alert §4: какие новые поля появились."""
        self._unknown_fields.setdefault(contract.name, set()).update(unknown)

    def _reject(
        self,
        contract: SourceContract,
        partition: Partition,
        location: str,
        reason: ReasonCode,
        detail: str,
    ) -> None:
        self._quarantine.add(
            reason,
            source=contract.name,
            raw_reference=location,
            partition=partition.canonical_path,
            detail=detail,
        )

    # ------------------------------------------------------------------ #
    # Вспомогательное
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record_id(contract: SourceContract, payload: Mapping[str, Any]) -> str:
        return KEY_SEPARATOR.join(str(payload[column]) for column in contract.primary_key)

    @staticmethod
    def _client_ref(contract: SourceContract, payload: Mapping[str, Any]) -> str | None:
        """Идентификатор клиента в терминах источника.

        Каноническим `client_id` он станет только в §7 — здесь это сырая
        ссылка, годная для отладки и для будущего разрешения identity.
        """
        if contract.client_field is None:
            return None
        value = payload.get(contract.client_field)
        return value if isinstance(value, str) else None

    def schema_alerts(self) -> dict[str, list[str]]:
        """Alert §4: источники, приславшие поля вне контракта.

        Возвращается отдельно от метрик, потому что §4 требует именно alert:
        `schema_violation_rate` покажет «стало хуже», но не скажет, какое
        поле появилось и в каком источнике.
        """
        return {
            source: sorted(fields) for source, fields in sorted(self._unknown_fields.items())
        }

    def merge(self, other: "SourceReader") -> None:
        """Присоединить alert-и воркера. Метрики и карантин сливаются своими
        объектами — здесь только то, что накапливает сам ридер."""
        for source, fields in other._unknown_fields.items():
            self._unknown_fields.setdefault(source, set()).update(fields)
