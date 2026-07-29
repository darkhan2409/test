"""Откуда берётся таблица identity mapping — §7, §30.

Здесь сторожатся две вещи, и обе появились после того, как conformance на
чистом клоне оказался зелёным по неправильной причине: путь к таблице был
относительным от текущего каталога, и прогон над временным набором читал
`data/raw` рабочей машины.

Первая — что каталог в конфиге снова не появится. Вторая — что на ENCODE
таблица берётся из замороженного состояния, и расхождение версий это заметит.
Вторая нужна именно потому, что таблица берётся из артефакта: §30 сверяет
пересчитанное состояние с замороженным, а раздел, взятый из самого артефакта,
совпадёт с собой при любом конфиге. Проверка версии — то, что осталось от §30
на этом участке, и без неё бы не осталось ничего.

Сам факт «прогон не читает ничего от текущего каталога» проверяется не здесь,
а в `test_conformance.py`: там для этого весь модуль уходит в пустой каталог.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.preprocessing.identity_resolver import (
    DatasetIdentityTable,
    FrozenIdentityTable,
    IdentityMappingConfig,
    IdentityMappingError,
    load_identity_mapping,
)
from src.preprocessing.schema import load_source_contracts

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "identity_mapping.yaml"
CONTRACTS = ROOT / "config" / "source_contracts.yaml"

# Единственный набор в git: `data/raw` не версионируется, и на чистом клоне
# его нет — именно поэтому тест, читавший оттуда, и оказался вырожденным.
GOLDEN_META = ROOT / "data" / "golden_input" / "_meta"


def policy(**overrides: object) -> dict:
    """Настоящая политика из `config/` с точечной правкой."""
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document.update(overrides)
    return document


@pytest.mark.parametrize(
    "table_file",
    [
        "data/raw/_meta/identity_mapping.json",  # ровно то, что было в конфиге
        "../golden_input/_meta/identity_mapping.json",
        "_meta/identity_mapping.json",
        "C:/data/identity_mapping.json",
    ],
)
def test_table_file_may_not_carry_a_directory(table_file: str):
    """Каталог в имени таблицы отклоняется (§7).

    Каталог — это и есть способ снова увести чтение из обрабатываемого набора
    в произвольное место файловой системы, где на одной машине файл лежит, а
    на другой нет.
    """
    with pytest.raises(ValueError, match="ожидалось имя файла без каталога"):
        IdentityMappingConfig.model_validate(policy(table_file=table_file))


def test_table_is_read_from_the_dataset_and_not_from_the_current_directory(
    tmp_path, monkeypatch
):
    """Читается таблица переданного набора, а не та, что лежит под ногами.

    Проверка **по содержимому**, а не по отсутствию файла: под текущим
    каталогом лежит своя таблица, по тому же пути `data/raw/_meta/`, каким
    код ходил до починки, и отличается она клиентами. Проверка «файла нет»
    была бы слабее ровно на рабочей машине — там файл есть, и именно поэтому
    дефект прожил так долго.

    Ожидаемое — литералы `C000000` и `C900000`, а не пересчёт по тем же
    правилам: сверяется код с ожиданием, а не две реализации.
    """
    dataset_meta = tmp_path / "dataset" / "_meta"
    dataset_meta.mkdir(parents=True)
    table = (GOLDEN_META / "identity_mapping.json").read_text(encoding="utf-8")
    (dataset_meta / "identity_mapping.json").write_text(table, encoding="utf-8")

    # Подсадная таблица: те же секции и ссылки, другие клиенты.
    decoy_meta = tmp_path / "cwd" / "data" / "raw" / "_meta"
    decoy_meta.mkdir(parents=True)
    (decoy_meta / "identity_mapping.json").write_text(
        table.replace('"C0', '"C9'), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path / "cwd")

    resolved = load_identity_mapping(
        CONFIG, load_source_contracts(CONTRACTS), table=DatasetIdentityTable(dataset_meta)
    )

    clients = resolved.clients()

    # Порядок проверок — не вкусовщина. Обе падают на этом дефекте, но
    # диагноз даёт только вторая: «клиента набора нет» одинаково звучит и
    # когда таблицу взяли не оттуда, и когда не взяли вовсе. Мутационный
    # прогон это и показал — объявленное сообщение не появлялось, потому что
    # первой срабатывала общая проверка.
    assert "C900000" not in clients, (
        "в результате клиент подсадной таблицы: путь к таблице снова считается "
        "от текущего каталога, а не от набора"
    )
    assert "C000000" in clients, "клиента набора нет — таблицу взяли не оттуда"


def test_frozen_table_refuses_a_version_that_drifted_from_the_config():
    """Версия в конфиге разошлась с замороженной — ENCODE останавливается (§30).

    Ожидаемое значение здесь — константа, а не пересчёт по тем же правилам:
    иначе сверялись бы две реализации одного правила, а не код с ожиданием.
    """
    registry = load_source_contracts(CONTRACTS)
    frozen = load_identity_mapping(
        CONFIG, registry, table=DatasetIdentityTable(GOLDEN_META)
    ).state()

    drifted = {**frozen, "identity_mapping_version": "0.0.1-drifted"}

    with pytest.raises(IdentityMappingError, match="не совпадает с замороженной"):
        load_identity_mapping(CONFIG, registry, table=FrozenIdentityTable(drifted))
