"""Канонический профиль клиента — §6.

Профиль всегда относится к моменту T: выбирается последний снимок с
`profile_time_utc <= T`, будущий запрещён. Если снимка нет вовсе, профиль всё
равно строится — из `MISSING` по обязательным полям — и помечается
`profile_snapshot_missing`. Пропустить клиента нельзя: у токенайзера блок
`[USR]` есть всегда.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import RESERVED_FIELD_NAMES
from .event import FieldValue, SourceMeta

UTC = timezone.utc


class CanonicalProfile(BaseModel):
    """Снимок профиля на cutoff T (§6, §32.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str = Field(min_length=1)
    profile_time_utc: datetime
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    source_meta: SourceMeta | None = None
    quality_flags: tuple[str, ...] = ()

    profile_snapshot_missing: bool = False
    """§6 п.3: снимка на T не нашлось, профиль собран из MISSING.

    Боковой quality-флаг: токенизируется только если явно объявлен полем
    профиля в Feature Schema.
    """

    @field_validator("profile_time_utc")
    @classmethod
    def _must_be_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("profile_time_utc должен быть с часовым поясом (aware)")
        return value.astimezone(UTC)

    @field_validator("fields")
    @classmethod
    def _fields_are_prepared(cls, value: dict[str, FieldValue]) -> dict[str, FieldValue]:
        """Те же ограничения, что и у события: никаких зарезервированных имён
        и никаких сырых значений."""
        for name, field_value in value.items():
            if name in RESERVED_FIELD_NAMES:
                raise ValueError(f"поле {name!r} недопустимо в профиле")
            if isinstance(field_value, str):
                if not field_value:
                    raise ValueError(f"поле {name!r}: пустая строка — используйте MISSING (§15.2)")
            else:
                if not field_value or any(not isinstance(x, str) or not x for x in field_value):
                    raise ValueError(f"поле {name!r}: значения — непустые строки")
        return value

    def is_before(self, cutoff: datetime) -> bool:
        """§6 п.1: снимок пригоден, только если он не позже T."""
        return self.profile_time_utc <= cutoff

    def to_output(self) -> dict[str, Any]:
        """Форма `prepared_profile` из §32.1."""
        return {
            "client_id": self.client_id,
            "profile_time_utc": self.profile_time_utc.isoformat().replace("+00:00", "Z"),
            "fields": dict(self.fields),
        }
