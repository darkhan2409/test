"""Каноническое событие — §5.

Модель проверяет все 13 правил раздела. Три из них стоит выделить, потому что
это не формальности, а то, ради чего раздел написан:

- `event_type` живёт top-level и запрещён внутри `fields` (пп. 3, 4, 8):
  токенайзер эмитит его первым после `[EVT]`, и дубль в полях дал бы два
  разных токена одного и того же факта;
- финальная `delta_from_previous_event` не хранится (п. 10): её считает
  токенайзер после truncation, и сохранённое значение относилось бы к
  событию, которого в окне уже нет;
- значения `fields` — только строки и списки строк: сырых чисел, `None` и
  `NaN` на выходе препроцессинга быть не может (§2.2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    DELTA_FIELD,
    EVENT_TYPE_FIELD,
    MAX_HOUR,
    MAX_WEEKDAY,
    MIN_HOUR,
    MIN_WEEKDAY,
    RESERVED_FIELD_NAMES,
)

UTC = timezone.utc

FieldValue = str | list[str]


class CalendarTimeFeatures(BaseModel):
    """Локальные календарные признаки — §5 п.9, §25.

    Считаются в бизнес-локальной зоне, а не из UTC (§12 п.4): в UTC+5 вечерняя
    покупка выглядит дневной, и поведенческий смысл признака теряется.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hour_of_day_local: int = Field(ge=MIN_HOUR, le=MAX_HOUR)
    day_of_week_local: int = Field(ge=MIN_WEEKDAY, le=MAX_WEEKDAY)


class SourceMeta(BaseModel):
    """Lineage записи — §5 п.11, §8. Не токенизируется."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_system: str
    source_record_id: str
    source_schema_version: str


class CanonicalEvent(BaseModel):
    """Нормализованное бизнес-событие (§5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    timestamp_utc: datetime
    calendar_timezone: str
    ordering_key: str = Field(min_length=1)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    calendar_time_features: CalendarTimeFeatures
    source_meta: SourceMeta | None = None
    quality_flags: tuple[str, ...] = ()

    lifetime_first: bool = False
    """§5 п.13 и §25.1: `true` только на самом раннем известном событии клиента.

    Нужен токенайзеру, чтобы отличить `FIRST_EVENT` от `WINDOW_START`, когда
    в production грузится не вся история, а недавний срез.
    """

    @field_validator("timestamp_utc")
    @classmethod
    def _must_be_aware_utc(cls, value: datetime) -> datetime:
        """§5 п.5: хранится абсолютный instant. Наивное время означает, что
        нормализация §12 не отработала, и сортировка событий поедет."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_utc должен быть с часовым поясом (aware)")
        return value.astimezone(UTC)

    @field_validator("calendar_timezone")
    @classmethod
    def _must_be_iana(cls, value: str) -> str:
        """§5 п.6, §12 п.5: именно IANA-зона, а не фиксированный offset.

        Фиксированный `+05:00` потерял бы историю: до 2024-03-01 Алматы жил
        в UTC+6, и §12.1 запрещает применять нынешнее смещение задним числом.
        """
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"{value!r} не является IANA timezone") from error
        return value

    @field_validator("fields")
    @classmethod
    def _fields_are_prepared(cls, value: dict[str, FieldValue]) -> dict[str, FieldValue]:
        """§5 пп. 4, 8, 10 и §2.2: что в `fields` быть не может."""
        for name in value:
            if name in RESERVED_FIELD_NAMES:
                reason = (
                    "дублировать event_type в fields запрещено (§5 п.4)"
                    if name == EVENT_TYPE_FIELD
                    else "финальную дельту считает токенайзер после truncation (§5 п.10)"
                )
                raise ValueError(f"поле {name!r} недопустимо в fields: {reason}")

        for name, field_value in value.items():
            if isinstance(field_value, str):
                if not field_value:
                    raise ValueError(f"поле {name!r}: пустая строка — используйте MISSING (§15.2)")
            else:
                if not field_value:
                    raise ValueError(f"поле {name!r}: пустой список значений")
                for item in field_value:
                    if not isinstance(item, str) or not item:
                        raise ValueError(
                            f"поле {name!r}: значения многозначного поля — непустые строки"
                        )
        return value

    def to_output(self, *, include_lineage: bool = False) -> dict[str, Any]:
        """Форма `prepared_events` из §32.2.

        `source_meta` и `quality_flags` по умолчанию не выводятся: §5 п.11
        запрещает их токенизировать без отдельного решения, и в контракте
        §32.2 их нет. Включаются явно, когда нужен lineage.
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc.isoformat().replace("+00:00", "Z"),
            "calendar_timezone": self.calendar_timezone,
            "ordering_key": self.ordering_key,
            "fields": dict(self.fields),
            "calendar_time_features": {
                "hour_of_day_local": self.calendar_time_features.hour_of_day_local,
                "day_of_week_local": self.calendar_time_features.day_of_week_local,
            },
        }
        if self.lifetime_first:
            payload["lifetime_first"] = True
        if include_lineage:
            payload["source_meta"] = (
                self.source_meta.model_dump() if self.source_meta is not None else None
            )
            payload["quality_flags"] = list(self.quality_flags)
        return payload
