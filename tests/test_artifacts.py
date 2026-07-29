"""Обязательные артефакты — §31.

Проверяется три вещи, и все три из тех, что не видно глазами.

**Перечень §31 не потерял пунктов.** Список продублирован здесь, как и
перечни §30 и §2: это единственное место, где он сверяется с регламентом, а
не сам с собой.

**Прогонные факты не попадают в хэш состояния.** `build timestamp`,
`train baselines` и происхождение кода описывают запуск, а не преобразование.
Попади время в `preprocessing_state_sha256` — два одинаковых BUILD дали бы
разные хэши, и сравнение single/multi-worker (§29 п.10) развалилось бы на
пустом месте. Ни один существующий тест этого не сторожит: раздел, добавленный
в состояние, выглядит как обычный раздел.

**Хэш кода описывает код.** `git commit` при грязном дереве называет не тот
код, который отработал; `code_state_sha256` считается по содержимому и врать
не может. Проверяется, что он реагирует на правку файла — иначе он ничего не
описывает.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from src.preprocessing.artifact_hasher import PreprocessingState
from src.preprocessing.artifacts import (
    SPEC_31_ARTIFACTS,
    ArtifactsError,
    RequiredArtifacts,
    code_provenance,
    require_clean_tree,
)

# Факты о **запуске**, а не о преобразовании. В состояние §30 им нельзя.
RUN_SPECIFIC = ("build_timestamp", "train_baselines", "baselines", "code_state_sha256",
                "code_commit_hash", "git_commit", "processing_time")


# --------------------------------------------------------------------------- #
# Перечень §31
# --------------------------------------------------------------------------- #


def test_spec_31_list_matches_the_regulation():
    """Тридцать два пункта §31 — дословно."""
    from_regulation = {
        "source_contracts", "identity_mapping", "event_mapping", "feature_schema",
        "closed_set_domains", "bucket_field_domains", "category_mappings", "dedup_config",
        "sessionization_config", "timestamp_policy", "calendar_timezone_policy",
        "cutoff_policy", "fx_normalization_config", "fx_max_staleness", "bucket_edges",
        "bucket_metadata", "time_delta_edges", "numeric_validation_rules",
        "high_cardinality_policy", "text_policy", "field_priorities", "max_values_per_field",
        "train_baselines", "deterministic_build_config", "hash_policy", "golden_vectors",
        "sampling_algorithm", "global_seed", "preprocessing_state_sha256",
        "code_commit_hash", "train_dataset_identifier", "build_timestamp",
    }

    assert len(SPEC_31_ARTIFACTS) == 32
    assert set(SPEC_31_ARTIFACTS) == from_regulation


def test_artifacts_cannot_be_built_with_a_missing_item():
    """Пункт нельзя не заполнить: `TypeError`, а не неполный набор артефактов."""
    values = {item.name: "x" for item in dataclass_fields(RequiredArtifacts)[1:]}

    with pytest.raises(TypeError):
        RequiredArtifacts(**values)


# --------------------------------------------------------------------------- #
# Граница между артефактом и состоянием
# --------------------------------------------------------------------------- #


def test_run_specific_facts_stay_out_of_the_state_hash():
    """Прогонные факты §31 не должны появиться среди разделов состояния §30.

    Критерий — участвует ли артефакт в преобразовании данных. `bucket_edges`
    участвуют и в хэше есть, хотя тоже посчитаны из TRAIN; `build timestamp`
    не участвует и меняется каждый запуск.

    Сторож нужен именно здесь: раздел, добавленный в `PreprocessingState`,
    выглядит как обычный раздел, а последствие — разные хэши у двух
    одинаковых BUILD и упавшая проверка §29 п.10.
    """
    sections = {item.name for item in dataclass_fields(PreprocessingState)}

    intruders = sorted(sections & set(RUN_SPECIFIC))

    assert not intruders, (
        "в хэшируемое состояние попали факты о прогоне: "
        + ", ".join(intruders)
        + " — два одинаковых BUILD дадут разные хэши, и сравнение "
        "single/multi-worker (§29 п.10) перестанет что-либо значить"
    )


# --------------------------------------------------------------------------- #
# Происхождение кода
# --------------------------------------------------------------------------- #


def test_code_hash_reacts_to_a_source_change(tmp_path: Path):
    """Хэш содержимого обязан меняться от правки исходника.

    Иначе он не описывает код, а только делает вид: артефакт говорил бы
    «собрано этим», не имея к этому отношения.
    """
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_bytes(b"value = 1\n")

    before = code_provenance(tmp_path)["code_state_sha256"]
    source.write_bytes(b"value = 2\n")
    after = code_provenance(tmp_path)["code_state_sha256"]

    assert before != after


def test_code_hash_ignores_files_that_do_not_run(tmp_path: Path):
    """Файл вне `src`/`tools`/`config` на хэш кода не влияет.

    Тесты и заметки прогон не выполняют, и их правка не меняет результат.
    Считать их значило бы получать новый хэш кода на каждую правку теста.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_bytes(b"value = 1\n")
    (tmp_path / "tests").mkdir()

    before = code_provenance(tmp_path)["code_state_sha256"]
    (tmp_path / "tests" / "test_x.py").write_bytes(b"assert True\n")
    after = code_provenance(tmp_path)["code_state_sha256"]

    assert before == after


def test_code_hash_without_any_source_is_blocking(tmp_path: Path):
    """Пустое дерево — не «хэш пустоты», а ошибка: описывать нечего."""
    with pytest.raises(ArtifactsError, match="нет ни одного файла кода"):
        code_provenance(tmp_path)


def test_dirty_tree_blocks_where_the_lie_is_expensive():
    """Грязное дерево блокирует заморозку эталона, а не обычный BUILD.

    §29.2 п.3 называет несовпадение golden-векторов блокирующей ошибкой
    релиза. Эталон, снятый с кода, которого нет ни в одном коммите,
    воспроизвести нельзя — и обнаружится это на чужой машине.
    """
    with pytest.raises(ArtifactsError, match="рабочее дерево грязное"):
        require_clean_tree({"dirty": True, "git_commit": "abc123"}, "заморозка эталона")

    assert require_clean_tree({"dirty": False, "git_commit": "abc123"}, "заморозка") is None
