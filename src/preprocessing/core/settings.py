"""Единый конфиг препроцессинга.

Главное здесь — разделение полей на две группы, и оно не косметическое:

- **доменная политика** (`POLICY_FIELDS`) входит в `preprocessing_state_sha256`
  (§30): cutoff, global_seed, окно устаревания курса, разрыв сессии, лимит
  многозначных полей. Изменение любого из них меняет результат обработки и
  обязано менять версию состояния;
- **окружение** (`ENVIRONMENT_FIELDS`) в хэш не входит: пути, число воркеров,
  debug. Иначе один и тот же датасет, посчитанный на двух машинах или на
  разном числе воркеров, дал бы разные хэши — ровно то, что §29 запрещает.

Забыть классифицировать новое поле легко, а последствие незаметное, поэтому
полнота классификации проверяется при импорте модуля.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

UTC = timezone.utc

# Поля, входящие в preprocessing_state_sha256 (§30).
POLICY_FIELDS: frozenset[str] = frozenset(
    {
        "cutoff_time",
        "global_seed",
        "fx_max_staleness_days",
        "session_gap_minutes",
        "max_values_per_field",
        "bucket_sample_size",
    }
)

# Поля окружения: на результат обработки не влияют, в хэш не входят.
ENVIRONMENT_FIELDS: frozenset[str] = frozenset(
    {
        "data_dir",
        "artifacts_dir",
        "config_dir",
        "workers",
        "debug",
    }
)


class PreprocessingSettings(BaseSettings):
    """Конфигурация прогона. Значения по умолчанию — из регламента."""

    model_config = SettingsConfigDict(
        env_prefix="PREP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    # --- доменная политика (входит в хэш состояния) ---

    cutoff_time: datetime = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
    """Cutoff T (§14). Единый глобальный на прогон."""

    global_seed: int = 20260101
    """Seed детерминированной выборки (§19.6, §29.1 п.12)."""

    fx_max_staleness_days: int = Field(default=3, ge=0)
    """Предельная давность курса (§18.2, базовое значение — 3 календарных дня)."""

    session_gap_minutes: int = Field(default=30, gt=0)
    """Разрыв, после которого начинается новая сессия (§20.1)."""

    max_values_per_field: int = Field(default=8, gt=0)
    """Лимит значений многозначного поля (§21, базовое значение — 8)."""

    bucket_sample_size: int = Field(default=50_000, gt=0)
    """Размер выборки на поле для расчёта границ бакетов (§19.6).

    Влияет на сами границы, поэтому это политика, а не окружение. На текущем
    объёме (десятки тысяч значений на поле) выборка ничего не отсекает —
    значение задано с запасом на боевые данные.
    """

    # --- окружение (в хэш НЕ входит) ---

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    config_dir: Path = Path("config")

    workers: int = Field(default=1, ge=1)
    """Число процессов обработки. Результат от него зависеть не должен (§29 п.10)."""

    debug: bool = False
    """Трассировка вход/выход компонентов. На результат не влияет."""

    @field_validator("cutoff_time")
    @classmethod
    def _cutoff_must_be_aware(cls, value: datetime) -> datetime:
        """Наивный cutoff означал бы неизвестную зону отсечки — и разный
        состав выборки на разных машинах."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cutoff_time должен быть с часовым поясом (например, ...Z)")
        return value.astimezone(UTC)

    # --- производные пути ---

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def golden_input_dir(self) -> Path:
        return self.data_dir / "golden_input"

    @property
    def prepared_dir(self) -> Path:
        return self.data_dir / "prepared"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    @property
    def debug_dir(self) -> Path:
        return self.data_dir / "debug"

    # --- производные значения политики ---

    @property
    def fx_max_staleness(self) -> timedelta:
        return timedelta(days=self.fx_max_staleness_days)

    @property
    def session_gap(self) -> timedelta:
        return timedelta(minutes=self.session_gap_minutes)

    def policy_state(self) -> dict[str, Any]:
        """Та часть конфига, что идёт в `preprocessing_state_sha256` (§30).

        Возвращает только JSON-совместимые значения: `datetime` уже приведён
        к строке, потому что каноническая сериализация запрещает неявные
        преобразования типов.
        """
        return {
            "bucket_sample_size": self.bucket_sample_size,
            "cutoff_time_utc": self.cutoff_time.isoformat().replace("+00:00", "Z"),
            "fx_max_staleness_days": self.fx_max_staleness_days,
            "global_seed": self.global_seed,
            "max_values_per_field": self.max_values_per_field,
            "session_gap_minutes": self.session_gap_minutes,
        }


def _check_field_classification() -> None:
    """Убедиться, что каждое поле отнесено ровно к одной группе.

    Неклассифицированное поле — тихая ошибка: политика не попала бы в хэш
    (и её изменение прошло бы незамеченным) либо окружение попало бы в хэш
    (и хэш стал бы машинозависимым).
    """
    declared = set(PreprocessingSettings.model_fields)
    classified = POLICY_FIELDS | ENVIRONMENT_FIELDS

    unclassified = sorted(declared - classified)
    if unclassified:
        raise RuntimeError(
            "поля конфига не отнесены ни к политике, ни к окружению: "
            + ", ".join(unclassified)
        )

    unknown = sorted(classified - declared)
    if unknown:
        raise RuntimeError("в классификации есть несуществующие поля: " + ", ".join(unknown))

    overlap = sorted(POLICY_FIELDS & ENVIRONMENT_FIELDS)
    if overlap:
        raise RuntimeError("поля отнесены сразу к двум группам: " + ", ".join(overlap))


_check_field_classification()

# Готового экземпляра здесь нет намеренно. Модульный синглтон не переживает
# `spawn`: воркер импортирует модуль заново и собирает свой — с умолчаниями,
# а не с тем, что настроил родитель. Разошлись бы при этом `POLICY_FIELDS`
# (`cutoff_time`, `global_seed`, `fx_max_staleness`, …), то есть ровно те
# поля, что входят в `preprocessing_state_sha256`; на тестах с настройками по
# умолчанию родитель и воркер совпали бы случайно, и расхождение не
# проявилось бы вовсе.
#
# Проверкой это не закрывается: проверять нечего, пока объект не используется,
# а в тот день, когда его импортируют, проверка уже опоздает. Поэтому запрет
# выражен отсутствием возможности — экземпляр создаёт вызывающий и передаёт
# его вниз явно, как и всё остальное, что уезжает в воркер.
