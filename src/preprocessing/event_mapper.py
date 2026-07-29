"""EventMapper — §8, §10.

Здесь сырая запись становится событием: получает утверждённый `event_type`
и стабильный `event_id`.

Два запрета определяют устройство компонента.

**§10 п.2 — неизвестный код не создаёт новый тип автоматически.** Поэтому
никакого fallback нет вовсе: ни `OTHER`, ни «взять код как есть». Список
типов закрыт конфигом, и код вне списка уходит в карантин. Негибкость
осознанная: молча появившийся `event_type` означает, что модель учится на
чём-то, чего никто не утверждал.

**§8 — `event_id` не использует случайность.** Ни UUID, ни текущее время, ни
номер воркера, ни номер строки. Пре-образ — источник, `source_record_id`,
нормализованный тип и время события, всё по `hash_policy` (§29.1). Повторный
прогон обязан дать тот же идентификатор, иначе не сойдутся ни golden-vectors,
ни сравнение single/multi-worker.

Не всякая запись становится событием (§10 п.5). Строки app-логов — сырьё для
`APP_SESSION` (§20), снимок профиля — материал §24, курс валют — справочник
§18. У них `event_type` пуст, и это не ошибка: их разбирают следующие
компоненты по типу источника.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core.debug_dump import DebugDump, Stage
from .core.hashing import event_id as make_event_id
from .core.monitor import DataQualityMonitor, Total
from .core.quarantine import Quarantine, ReasonCode
from .records import TimedRecord
from .schema.source_contract import SourceContractRegistry

COMPONENT = "event_mapper"


class EventMappingError(RuntimeError):
    """Ошибка маппинга событий — блокирующая."""


@dataclass(frozen=True)
class MappedRecord(TimedRecord):
    """Запись с утверждённым типом события и стабильным `event_id` (§8, §10)."""

    event_type: str | None = None
    """`None` — запись не становится модельным событием (§10 п.5): строка
    app-лога до сессионизации, снимок профиля, строка справочника курсов."""

    event_id: str | None = None
    """Заполнен ровно тогда, когда заполнен `event_type`: у не-события нет
    идентификатора события."""

    def debug_row(self) -> dict[str, Any]:
        return {**super().debug_row(), "event_type": self.event_type, "event_id": self.event_id}

    def lineage(self) -> dict[str, Any]:
        """Цепочка §8: событие → сырая запись → файл/партиция → версия схемы."""
        return {
            "event_id": self.event_id,
            "source_system": self.source,
            "source_record_id": self.source_record_id,
            "source_partition": self.partition,
            "source_line": self.line_number,
            "source_schema_version": self.source_schema_version,
        }


class SourceEventMapping(BaseModel):
    """Маппинг кодов одного источника."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code_field: str | None
    """Колонка с кодом операции. `None` — у источника нет кодов, тип задаётся
    целиком через `default_event_type`."""

    codes: dict[str, str | None] = Field(default_factory=dict)
    """`код → тип`. Значение `null` — утверждённый технический message,
    который модельным событием не становится (§10 п.5). Это не то же самое,
    что неизвестный код: неизвестный уходит в карантин."""

    default_event_type: str | None = None

    @model_validator(mode="after")
    def _shape_matches_source(self) -> "SourceEventMapping":
        if self.code_field is None and self.codes:
            raise ValueError("коды заданы, но не указано поле кода")
        if self.code_field is not None and not self.codes:
            raise ValueError(f"поле кода {self.code_field!r} задано, но список кодов пуст")
        if self.code_field is not None and self.default_event_type is not None:
            # Иначе неизвестный код тихо получал бы тип по умолчанию —
            # ровно то автосоздание, которое запрещает §10 п.2.
            raise ValueError(
                "default_event_type вместе с code_field: неизвестный код получил бы "
                "тип по умолчанию вместо карантина (§10 п.2)"
            )
        return self


class EventMapping(BaseModel):
    """Версионируемый маппинг событий (§10 п.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_mapping_version: str = Field(min_length=1)
    event_types: tuple[str, ...] = Field(min_length=1)
    sources: dict[str, SourceEventMapping] = Field(min_length=1)

    @model_validator(mode="after")
    def _types_are_approved(self) -> "EventMapping":
        approved = set(self.event_types)
        if len(approved) != len(self.event_types):
            raise ValueError("список event_types содержит повторы")

        for name in sorted(self.sources):
            mapping = self.sources[name]
            targets = {value for value in mapping.codes.values() if value is not None}
            if mapping.default_event_type is not None:
                targets.add(mapping.default_event_type)
            unknown = sorted(targets - approved)
            if unknown:
                raise ValueError(
                    f"{name}: типы вне утверждённого списка: {', '.join(unknown)} (§10 п.2)"
                )
        return self

    def state(self) -> dict[str, Any]:
        """Состояние для §30."""
        return self.model_dump(mode="json")


def load_event_mapping(path: Path, registry: SourceContractRegistry) -> EventMapping:
    """Загрузить маппинг и проверить его против Source Contracts."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise EventMappingError(f"{path}: ожидался YAML-объект")
    mapping = EventMapping.model_validate(document)

    missing = sorted(set(registry.sources) - set(mapping.sources))
    if missing:
        raise EventMappingError("нет маппинга событий для источников: " + ", ".join(missing))
    extra = sorted(set(mapping.sources) - set(registry.sources))
    if extra:
        raise EventMappingError("маппинг описывает несуществующие источники: " + ", ".join(extra))

    for name in sorted(mapping.sources):
        code_field = mapping.sources[name].code_field
        if code_field is None:
            continue
        contract = registry.contract(name)
        if code_field not in contract.columns:
            raise EventMappingError(
                f"{name}: поля кода {code_field!r} нет в схеме источника"
            )
        if not contract.columns[code_field].required:
            raise EventMappingError(
                f"{name}: поле кода {code_field!r} не гарантировано контрактом — "
                "тип события не должен зависеть от необязательного поля"
            )

    return mapping


class EventMapper:
    """Присвоение утверждённого типа и стабильного идентификатора."""

    def __init__(
        self,
        registry: SourceContractRegistry,
        mapping: EventMapping,
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
        self._seen_event_ids: dict[str, str] = {}

    def map(self, records: Iterable[TimedRecord]) -> Iterator[MappedRecord]:
        tracing = self._debug.enabled

        for record in records:
            if tracing:
                self._debug.record(COMPONENT, Stage.IN, [record.debug_row()])

            mapped = self._map_one(record)
            if mapped is None:
                continue
            if tracing:
                self._debug.record(COMPONENT, Stage.OUT, [mapped.debug_row()])
            yield mapped

    def _map_one(self, record: TimedRecord) -> MappedRecord | None:
        self._monitor.add_total(Total.RECORDS_MAPPED)
        mapping = self.mapping.sources[record.source]

        if mapping.code_field is None:
            return self._build(record, mapping.default_event_type)

        code = record.payload.get(mapping.code_field)
        if not isinstance(code, str) or code not in mapping.codes:
            # §10 п.3: неизвестный тип — карантин и метрика. Создать новый
            # `event_type` на лету запрещено (§10 п.2), поэтому альтернативы
            # карантину здесь нет ни одной.
            self._quarantine.add(
                ReasonCode.UNKNOWN_EVENT_TYPE,
                source=record.source,
                raw_reference=record.raw_reference,
                partition=record.partition,
                detail=(
                    f"{mapping.code_field}={code!r} нет в маппинге "
                    f"версии {self.mapping.event_mapping_version}"
                ),
            )
            return None

        return self._build(record, mapping.codes[code])

    def _build(self, record: TimedRecord, event_type: str | None) -> MappedRecord:
        identifier = None
        if event_type is not None:
            identifier = make_event_id(
                source_system=record.source,
                source_record_id=record.source_record_id,
                event_type=event_type,
                event_timestamp=record.timestamp_utc,
            )
            self._check_unique(identifier, record)

        return MappedRecord(
            source=record.source,
            partition=record.partition,
            line_number=record.line_number,
            source_record_id=record.source_record_id,
            source_schema_version=record.source_schema_version,
            client_ref=record.client_ref,
            payload=record.payload,
            client_id=record.client_id,
            timestamp_utc=record.timestamp_utc,
            calendar_timezone=record.calendar_timezone,
            processing_time_utc=record.processing_time_utc,
            quality_flags=record.quality_flags,
            event_type=event_type,
            event_id=identifier,
        )

    def _check_unique(self, identifier: str, record: TimedRecord) -> None:
        """Один `event_id` на одно событие.

        Совпадение означало бы либо пропущенный дубль (§9), либо коллизию
        пре-образа. Оба случая ниже по цепочке выглядят как «событие
        потерялось», и найти причину там уже нечем.
        """
        previous = self._seen_event_ids.get(identifier)
        if previous is not None and previous != record.raw_reference:
            raise EventMappingError(
                f"event_id {identifier} у двух записей: {previous} и {record.raw_reference}"
            )
        self._seen_event_ids[identifier] = record.raw_reference
