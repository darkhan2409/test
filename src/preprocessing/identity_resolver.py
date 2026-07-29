"""IdentityResolver — §7.

Источники называют клиента по-разному: `client_ref`, `cardholder_id`, `cif`,
`login_id`. Модель обязана видеть одного клиента, поэтому все ссылки сводятся
к каноническому `client_id`.

Главное правило §7 — «неоднозначный mapping запрещено разрешать случайно».
Случайность здесь возникает тише, чем кажется: `json.loads` на дубликате
ключа молча оставляет последнее значение, и таблица с конфликтом выглядит
исправной. Поэтому таблица разбирается с проверкой пар, а не словарём.

Что компонент **не** делает: не создаёт `client_id` для неизвестной ссылки и
не угадывает клиента по соседним полям. Неизвестная ссылка — карантин (§34),
и это не потеря данных, а отказ выдумывать связь.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .core.debug_dump import DebugDump, Stage
from .core.monitor import DataQualityMonitor
from .core.quarantine import Quarantine, ReasonCode
from .records import IdentifiedRecord, SourceRecord
from .schema.source_contract import PiiClass, SourceContractRegistry, SourceKind

COMPONENT = "identity_resolver"


class IdentityMappingError(RuntimeError):
    """Ошибка таблицы соответствий — блокирующая.

    Битая identity-таблица не «портит несколько записей»: она тихо смешивает
    истории разных клиентов, а это худший вид ошибки в этой модели.
    """


class IdentityMappingConfig(BaseModel):
    """Политика identity resolution (§7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_mapping_version: str = Field(min_length=1)
    table_path: Path
    client_id_pattern: str = Field(min_length=1)


@dataclass(frozen=True)
class IdentityMapping:
    """Проверенная таблица «ссылка источника → client_id»."""

    version: str
    sections: dict[str, dict[str, str]]
    """Ключ секции — `<источник>.<client_field>`. Пространства имён у разных
    источников независимы: `000005` в одном источнике и `000005` в другом —
    разные ссылки, а не конфликт."""

    def resolve(self, section: str, reference: str) -> str | None:
        return self.sections.get(section, {}).get(reference)

    def clients(self) -> set[str]:
        return {client for section in self.sections.values() for client in section.values()}

    def state(self) -> dict[str, Any]:
        """Состояние для §30. Сама таблица входит целиком: смена связи
        клиента меняет результат обработки и обязана менять хэш."""
        return {
            "identity_mapping_version": self.version,
            "sections": {
                name: dict(sorted(values.items())) for name, values in sorted(self.sections.items())
            },
        }


def section_name(source: str, client_field: str) -> str:
    return f"{source}.{client_field}"


def load_identity_mapping(
    config_path: Path, registry: SourceContractRegistry, *, base_dir: Path = Path(".")
) -> IdentityMapping:
    """Загрузить политику и таблицу, проверив их против Source Contracts."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise IdentityMappingError(f"{config_path}: ожидался YAML-объект")
    config = IdentityMappingConfig.model_validate(document)

    _check_client_fields_are_not_pii(registry)

    table_path = base_dir / config.table_path
    if not table_path.exists():
        raise IdentityMappingError(f"нет таблицы identity mapping: {table_path}")

    raw = json.loads(table_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    sections = _check_sections(raw, registry, re.compile(config.client_id_pattern))

    return IdentityMapping(version=config.identity_mapping_version, sections=sections)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Не дать повторяющемуся ключу молча перезаписать предыдущий.

    Стандартный разбор JSON оставляет последнее значение. Для identity-таблицы
    это ровно «конфликтующая связь разрешилась случайно» — то, что §7 п.3
    запрещает.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise IdentityMappingError(
                f"ключ {key!r} встречается в таблице дважды: {seen[key]!r} и {value!r} — "
                "конфликтующая связь (§7 п.3)"
            )
        seen[key] = value
    return seen


def _check_client_fields_are_not_pii(registry: SourceContractRegistry) -> None:
    """§7 п.4: PII не может быть идентификатором клиента в пайплайне."""
    for name in sorted(registry.sources):
        contract = registry.sources[name]
        if contract.client_field is None:
            continue
        column = contract.columns[contract.client_field]
        if column.pii is PiiClass.DIRECT_IDENTIFIER:
            raise IdentityMappingError(
                f"{name}: поле клиента {contract.client_field!r} классифицировано как "
                "direct_identifier — PII не может быть идентификатором (§7 п.4)"
            )


def _check_sections(
    raw: Any, registry: SourceContractRegistry, client_id_pattern: re.Pattern[str]
) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise IdentityMappingError("таблица identity mapping должна быть объектом")

    expected = {
        section_name(name, contract.client_field): name
        for name, contract in registry.sources.items()
        if contract.client_field is not None
    }

    missing = sorted(set(expected) - set(raw))
    if missing:
        raise IdentityMappingError(
            "в таблице нет секций для источников: " + ", ".join(missing)
        )
    extra = sorted(set(raw) - set(expected))
    if extra:
        raise IdentityMappingError(
            "в таблице есть секции без источника в Source Contracts: " + ", ".join(extra)
        )

    sections: dict[str, dict[str, str]] = {}
    for section in sorted(expected):
        values = raw[section]
        if not isinstance(values, dict):
            raise IdentityMappingError(f"секция {section}: ожидался объект")

        owners: dict[str, str] = {}
        for reference, client_id in values.items():
            if not isinstance(reference, str) or not reference:
                raise IdentityMappingError(f"секция {section}: пустая ссылка источника")
            if not isinstance(client_id, str) or not client_id_pattern.fullmatch(client_id):
                raise IdentityMappingError(
                    f"секция {section}: {client_id!r} не подходит под шаблон "
                    f"{client_id_pattern.pattern} — идентификатор модели обязан быть "
                    "непрозрачным (§7 п.4)"
                )
            if client_id in owners:
                # Две ссылки одного источника на одного клиента — это merge,
                # а §7 п.6 требует, чтобы merge был исторически корректен
                # относительно T. Такой политики в конфиге нет, поэтому
                # молча склеивать нельзя.
                raise IdentityMappingError(
                    f"секция {section}: {client_id} привязан к двум ссылкам "
                    f"({owners[client_id]}, {reference}) — это merge клиентов, "
                    "который §7 п.6 требует описать политикой относительно T"
                )
            owners[client_id] = reference
        sections[section] = dict(sorted(values.items()))

    return sections


class IdentityResolver:
    """Приведение ссылок источников к каноническому `client_id`."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        mapping: IdentityMapping,
        *,
        monitor: DataQualityMonitor,
        quarantine: Quarantine,
        debug: DebugDump | None = None,
    ) -> None:
        self.registry = registry
        self.mapping = mapping
        self._monitor = monitor
        self._quarantine = quarantine
        self._debug = debug or DebugDump(enabled=False, debug_dir=Path("."))

    def resolve(self, records: Iterable[SourceRecord]) -> Iterator[IdentifiedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            resolved = self._resolve_one(record)
            if resolved is None:
                continue
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [resolved.debug_row()])
            yield resolved

    def _resolve_one(self, record: SourceRecord) -> IdentifiedRecord | None:
        contract = self.registry.contract(record.source)

        if contract.client_field is None:
            # Справочный источник: клиента нет по построению, и это не повод
            # для карантина (§34 говорит про «невозможно определить», а не
            # про «не предусмотрен»).
            if contract.kind is not SourceKind.REFERENCE:
                raise IdentityMappingError(
                    f"{record.source}: нет client_field у источника {contract.kind}"
                )
            return _identified(record, None)

        reference = record.client_ref
        if not reference:
            self._reject(record, "ссылка на клиента пуста или отсутствует")
            return None

        client_id = self.mapping.resolve(
            section_name(contract.name, contract.client_field), reference
        )
        if client_id is None:
            self._reject(record, f"ссылки {reference!r} нет в таблице identity mapping")
            return None

        return _identified(record, client_id)

    def _reject(self, record: SourceRecord, detail: str) -> None:
        """В карантин. §33 не описывает identity resolution, поэтому своей
        метрики у причины нет — считается общим счётчиком карантина."""
        self._quarantine.add(
            ReasonCode.UNRESOLVED_CLIENT_ID,
            source=record.source,
            raw_reference=record.raw_reference,
            partition=record.partition,
            detail=detail,
        )


def _identified(record: SourceRecord, client_id: str | None) -> IdentifiedRecord:
    return IdentifiedRecord(
        source=record.source,
        partition=record.partition,
        line_number=record.line_number,
        source_record_id=record.source_record_id,
        source_schema_version=record.source_schema_version,
        client_ref=record.client_ref,
        payload=record.payload,
        client_id=client_id,
    )
