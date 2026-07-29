"""Реестр версий артефактов — §30.

Регламент требует единый комплект версий: если хоть одна не зафиксирована,
нельзя ни посчитать `preprocessing_state_sha256`, ни проверить совместимость
при загрузке. Поэтому здесь перечислены ровно те имена, что в §30, а
незаполненные помечены `UNSET` и ловятся `require_complete()` — молчаливая
пустая версия хуже отсутствующей.

Версии, которыми владеет код (политика хэширования и формат сериализации),
подставляются из соответствующих модулей. Остальные появляются по мере
заморозки артефактов на этапах 2–3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .canonical import SERIALIZATION_CONFIG_VERSION
from .hashing import HASH_POLICY_VERSION

UNSET = "UNSET"


class VersionsIncompleteError(ValueError):
    """Не все версии зафиксированы — блокирующая ошибка перед freeze и загрузкой."""


class PreprocessingVersions(BaseModel):
    """Единый комплект версий (§30) плюс content hash состояния."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- перечень §30, в порядке регламента ---
    source_contract_version: str = UNSET
    identity_mapping_version: str = UNSET
    event_mapping_version: str = UNSET
    feature_schema_version: str = UNSET
    closed_set_domains_version: str = UNSET
    bucket_field_domains_version: str = UNSET
    category_mapping_version: str = UNSET
    timestamp_policy_version: str = UNSET
    calendar_timezone_policy_version: str = UNSET
    dedup_policy_version: str = UNSET
    sessionization_version: str = UNSET
    fx_normalization_version: str = UNSET
    bucket_edges_version: str = UNSET
    time_delta_edges_version: str = UNSET
    text_policy_version: str = UNSET
    hash_policy_version: str = HASH_POLICY_VERSION
    golden_vectors_version: str = UNSET
    preprocessing_version: str = UNSET

    # §30 перечисляет `fx_max_staleness` в одном списке с версиями, хотя это
    # значение, а не версия. Живёт оно в настройках (§18.2); здесь дублируется
    # ровно для того, чтобы попасть в метаданные выхода вместе с версиями.
    fx_max_staleness_days: int | None = None

    # Не входит в перечень §30, но требуется §29.1 п.4: выбранный формат float
    # версионируется отдельно, иначе его смена пройдёт незамеченной.
    serialization_config_version: str = SERIALIZATION_CONFIG_VERSION

    # Заполняется ArtifactHasher (шаг 3.1), не при создании реестра.
    preprocessing_state_sha256: str | None = None

    def unset_fields(self) -> list[str]:
        """Версии, которые ещё не зафиксированы."""
        missing = [
            name
            for name, value in self.model_dump().items()
            if isinstance(value, str) and value == UNSET
        ]
        if self.fx_max_staleness_days is None:
            missing.append("fx_max_staleness_days")
        return sorted(missing)

    def require_complete(self) -> None:
        """Проверить комплектность перед freeze (§27) и перед загрузкой (§28)."""
        missing = self.unset_fields()
        if missing:
            raise VersionsIncompleteError(
                "не зафиксированы версии: " + ", ".join(missing)
            )
        if not self.preprocessing_state_sha256:
            raise VersionsIncompleteError("не рассчитан preprocessing_state_sha256 (§30)")

    def as_metadata(self) -> dict[str, Any]:
        """Блок версий для метаданных выхода (§32.3)."""
        return self.model_dump(exclude_none=True)
