"""Конфигурация генератора синтетических данных."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

UTC = timezone.utc


class DatasetRole(StrEnum):
    """Для чего датасет предназначен.

    Роль публикуется в манифесте и решает, что с набором можно делать: BUILD
    (§27) выполняется только на TRAIN, а golden-набор проходит ENCODE на уже
    замороженных артефактах (§29.2). Разбиений Validation/Test генератор пока
    не делает — их роли здесь нет, потому что объявить набор, которого не
    существует, значило бы обещать разбиение.
    """

    TRAIN = "train"
    GOLDEN_INPUT = "golden_input"


class GeneratorConfig(BaseModel):
    """Полностью определяет выход генератора: один и тот же конфиг → те же файлы байт-в-байт.

    `cutoff_time` генератору нужен не для фильтрации (обрезает препроцессинг, §14),
    а для того, чтобы осознанно раскладывать события до/после T и по границе T.
    Значение обязано совпадать с T пайплайна.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 20260101

    n_clients: int = Field(default=200, ge=1)

    # История намеренно начинается до 2024-03-01: до этой даты в Казахстане
    # действовали разные региональные смещения, и §12.1 запрещает ретроактивно
    # применять единый UTC+05 к старым событиям.
    history_start: date = date(2023, 9, 1)
    history_end: date = date(2026, 2, 28)  # включительно

    cutoff_time: datetime = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)

    # Множитель числа «фоновых» событий. Краевые случаи инъектируются поверх и
    # от него не зависят: golden-набор должен быть маленьким, но полным.
    volume_scale: float = Field(default=1.0, gt=0.0)

    debug_clean_clients: int = Field(default=2, ge=0)
    """Сколько клиентов без единого краевого случая датасет обещает трассировке.

    Число публикуется в манифесте набора и оттуда его читает debug-режим
    препроцессинга: на демо главный экран показывает путь нормальной
    транзакции, а дамп из одного брака показывать нечего.

    Живёт оно здесь, а не константой в debug-режиме, по двум причинам. Состав
    клиентов знает только генератор — от этого числа зависит размер
    golden-набора (`MIN_CLIENTS` ролей плюс столько чистых). И у разных
    наборов оно может отличаться: константа в коде этого не выразит, а поле
    манифеста — да.
    """

    out_dir: Path = Path("data/raw")
    dataset_name: str = "main"
    dataset_role: DatasetRole = DatasetRole.TRAIN
    generator_version: str = "0.1.0"
    # 1.1.0 — состав golden-набора изменился: к 12 ролевым клиентам добавлены
    # два «чистых», без единого краевого случая (§29.2 требует версионировать
    # набор, а изменение состава без смены версии сделало бы два разных набора
    # неразличимыми по манифесту).
    golden_vectors_version: str = "1.1.0"

    @model_validator(mode="after")
    def _check_period(self) -> "GeneratorConfig":
        if self.history_start >= self.history_end:
            raise ValueError("history_start должен быть раньше history_end")
        cutoff_date = self.cutoff_time.astimezone(UTC).date()
        if not (self.history_start < cutoff_date < self.history_end):
            raise ValueError(
                "cutoff_time должен попадать внутрь периода истории, "
                "иначе не получится сгенерировать события после T"
            )
        return self
