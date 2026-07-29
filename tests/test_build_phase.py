"""Допуск набора к BUILD — §27, последний абзац.

«BUILD PHASE не выполняется на отдельном примере, batch, Validation, Test или
inference». Пример и батч в `run_build` передать нечем — параметра нет. А вот
не тот набор передать можно, и решает это `TrainDataset`: он собирается только
из набора, объявившего роль `train`. Дальше роль гарантирует тип, и проверять
её повторно негде.

Проверяется именно отказ: успешный случай виден в полном прогоне BUILD, а
молчаливое согласие на golden-набор не видно вообще ничем — артефакты
получились бы, просто посчитанные не на том.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.preprocessing.pipeline import BuildPhaseError, TrainDataset

TRAIN_MANIFEST = {
    "dataset_name": "main",
    "dataset_role": "train",
    "golden_vectors_version": "1.1.0",
}


def dataset(tmp_path: Path, manifest: dict | None) -> Path:
    root = tmp_path / "raw"
    (root / "_meta").mkdir(parents=True)
    if manifest is not None:
        (root / "_meta" / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
    return root


def test_train_dataset_is_accepted(tmp_path: Path):
    """Набор с ролью `train` даёт объект с идентификатором для §31."""
    loaded = TrainDataset.load(dataset(tmp_path, TRAIN_MANIFEST))

    assert loaded.identifier == "main"


def test_golden_input_is_refused(tmp_path: Path):
    """Golden-набор проходит ENCODE на замороженных артефактах, не BUILD."""
    manifest = {**TRAIN_MANIFEST, "dataset_name": "golden_input", "dataset_role": "golden_input"}

    with pytest.raises(BuildPhaseError, match="только на 'train'"):
        TrainDataset.load(dataset(tmp_path, manifest))


def test_dataset_without_role_is_refused(tmp_path: Path):
    """Набор, не объявивший роль, не считается TRAIN по умолчанию.

    Умолчание здесь означало бы «раз не сказано, значит можно» — и BUILD
    однажды посчитал бы границы по Validation.
    """
    manifest = {key: value for key, value in TRAIN_MANIFEST.items() if key != "dataset_role"}

    with pytest.raises(BuildPhaseError, match="роль набора"):
        TrainDataset.load(dataset(tmp_path, manifest))


def test_dataset_without_manifest_is_refused(tmp_path: Path):
    """Без манифеста роль набора неизвестна, а не «вероятно train»."""
    with pytest.raises(BuildPhaseError, match="нет манифеста"):
        TrainDataset.load(dataset(tmp_path, None))


def test_dataset_without_identifier_is_refused(tmp_path: Path):
    """§31 требует хранить идентификатор TRAIN-датасета.

    Артефакт без него не отвечает на вопрос, на чём посчитан, — а это
    единственное, зачем идентификатор нужен.
    """
    manifest = {key: value for key, value in TRAIN_MANIFEST.items() if key != "dataset_name"}

    with pytest.raises(BuildPhaseError, match="dataset_name"):
        TrainDataset.load(dataset(tmp_path, manifest))
