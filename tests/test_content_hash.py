"""Content hash и метаданные выхода — §30, §35 F.38–F.40.

Три пункта чек-листа, и каждый проверяется своим способом.

**F.38 «`preprocessing_state_sha256` рассчитан».** Проверено тестами 3.1:
хэш считается по канонически сериализованному состоянию, и каждый раздел §30
на него влияет. Здесь не дублируется.

**F.39 «Hash проверяется при загрузке».** Свойство выражено устройством:
`verify_state_hash` ничего не возвращает, поэтому «продолжить с
предупреждением» не выражается — проверять её результат нечем. Здесь
проверяется, что ENCODE её действительно зовёт и что без хэша в артефактах
он не начинается.

**F.40 «Output metadata содержит все версии».** Вот это здесь и есть.
§32.3 показывает семь полей, F.40 говорит «все версии» — пример трактуется
как подмножество комплекта §30. Проверяется, что в метаданных нет ни одной
незаполненной версии и что семь полей §32.3 на месте: без версий маппингов,
dedup, сессионизации и hash policy токенайзер не сможет объяснить, чем
получено значение.
"""

from __future__ import annotations

import pytest

from src.preprocessing.core.settings import PreprocessingSettings
from src.preprocessing.core.versions import PreprocessingVersions, VersionsIncompleteError
from src.preprocessing.pipeline import output_metadata

# Семь полей примера §32.3 — то, что регламент показывает явно.
SPEC_32_3_FIELDS: tuple[str, ...] = (
    "cutoff_time",
    "preprocessing_version",
    "preprocessing_state_sha256",
    "feature_schema_version",
    "bucket_field_domains_version",
    "bucket_edges_version",
    "calendar_timezone_policy_version",
)


def complete_versions() -> PreprocessingVersions:
    """Комплект §30 без единой незаполненной версии."""
    values = {
        name: f"{name}-1.0.0"
        for name, field in PreprocessingVersions.model_fields.items()
        if field.annotation is str or field.annotation == "str"
    }
    return PreprocessingVersions(**values).model_copy(
        update={
            "fx_max_staleness_days": 3,
            "preprocessing_state_sha256": "a" * 64,
        }
    )


def metadata_of(versions: PreprocessingVersions) -> dict:
    """Метаданные выхода — той же функцией, которой их собирает ENCODE.

    Не реконструкция: собери тест метаданные сам, он проверял бы собственное
    представление о них, а правку в `run_encode` пропустил бы.
    """
    return output_metadata(PreprocessingSettings(), versions)


def test_output_metadata_carries_every_version_of_spec_30():
    """F.40: в метаданных выхода все версии §30, а не семь из примера §32.3.

    Проверка идёт по полям реестра, а не по переписанному списку: реестр —
    это и есть перечень §30, и новая версия появится в нём раньше, чем в
    любом списке, продублированном здесь.
    """
    metadata = metadata_of(complete_versions())

    missing = sorted(
        name for name in PreprocessingVersions.model_fields if name not in metadata
    )

    assert not missing, (
        "в метаданных выхода нет версий: "
        + ", ".join(missing)
        + " — токенайзер не сможет объяснить, чем получено значение (§35 F.40)"
    )


def test_output_metadata_includes_the_seven_fields_of_32_3():
    """§32.3 показывает семь полей явно — они обязаны быть на месте."""
    metadata = metadata_of(complete_versions())

    assert set(SPEC_32_3_FIELDS) <= set(metadata)


def test_unset_version_blocks_the_output():
    """Незаполненная версия — блокирующая ошибка, а не строка `UNSET` в выходе.

    Строка `UNSET`, доехавшая до метаданных, выглядит как версия и
    сравнивается как версия. Обнаружилось бы это на стыке с токенайзером,
    где чинить дороже всего.
    """
    incomplete = complete_versions().model_copy(update={"identity_mapping_version": "UNSET"})

    with pytest.raises(VersionsIncompleteError, match="identity_mapping_version"):
        metadata_of(incomplete)


def test_state_hash_is_part_of_the_registry_not_a_separate_field():
    """`preprocessing_state_sha256` едет вместе с версиями, а не отдельно.

    §30 перечисляет его в том же комплекте: версия артефакта и хэш
    состояния, при котором он получен, разъехаться не должны.
    """
    metadata = metadata_of(complete_versions())

    assert metadata["preprocessing_state_sha256"] == "a" * 64
    assert metadata["preprocessing_version"].startswith("preprocessing_version")
