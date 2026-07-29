"""Conformance: §29.2 golden-vectors и §29 п.10 single/multi-worker.

Обе проверки раньше жили в скриптах вне `tests/` — то есть не запускались
никогда, кроме как руками. §29.2 п.5 требует прогонять golden перед каждым
релизом, и «проверка написана» этого не заменяет.

**Почему они отдельным маркером.** Здесь поднимается генератор, полный BUILD
и четыре процесса — около минуты, а не секунды. Тест, из-за которого
перестают гонять `pytest`, защищает хуже отсутствующего. Обычный прогон их
пропускает, запуск явный:

    pytest -m conformance -p no:faulthandler

`-p no:faulthandler` — из-за Windows: при `CreateProcess` спавна ОС бросает
first-chance access violation, faulthandler печатает её стек, и вывод
выглядит аварийным при зелёном результате.

**Почему всё пересобирается, а не берётся из рабочего каталога.** `data/` и
`artifacts/` не в git: у артефактов своя версионность по §30 и §31, и git
завёл бы вторую. Значит на свежем клоне их нет вовсе. Но и не нужно:
генератор и BUILD детерминированы — это доказано отдельно, — поэтому вся
цепочка от `seed` до эталона восстанавливается из закоммиченных конфигов.
Совпадение проверяется по `preprocessing_state_sha256` из самого эталона:
если пересобранное состояние совпало с замороженным, промежуточное хранить
было незачем.

Единственное, что берётся с диска, — `golden_input` и `golden_expected`:
они в git, потому что §29.2 требует именно **замороженный** набор. Пересобери
их тест сам, он сравнивал бы прогон с прогоном.

**Почему прогон уходит в пустой каталог.** Всё написанное выше однажды было
неправдой, и заметить это здесь было нельзя. Таблицу identity mapping код брал
по относительному пути от текущего каталога — то есть из `data/raw` рабочей
машины. На машине разработчика файл есть всегда, поэтому прогон был зелёным, а
пересборка шла не от `seed`, а от `seed` плюс файл, которого на чистом клоне
нет. Мутационная проверка такое не ловит по построению: любая мутация зелёная,
пока недостающий вход молча подставляет рабочая машина.

Поэтому `isolated_cwd` уводит весь модуль в пустой временный каталог. Там нет
ни `data/`, ни `config/`, ни `artifacts/`, и подставить оттуда нечего:
относительный путь, вычисленный от текущего каталога, падает на любой машине, а
не только на чистом клоне. Всё, что модулю действительно нужно, адресуется от
`ROOT` — то есть от репозитория, а не от места запуска.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.generator.build_dataset import build as build_dataset
from src.generator.config import GeneratorConfig
from src.preprocessing.core.settings import PreprocessingSettings
from src.preprocessing.parallel import ParallelismNotProvenError
from src.preprocessing.golden import build_expected, compare_expected, load_expected
from src.preprocessing.pipeline import (
    Dataset,
    FrozenArtifacts,
    TrainDataset,
    compare_workers,
    compare_workers_encode,
    freeze_build,
    run_build,
    run_encode,
)

pytestmark = pytest.mark.conformance


def conducted(experiment):
    """Провести опыт §29 п.10, различая два его исхода.

    «Не пройдена» — выходы разошлись, это ошибка и красный прогон.
    «Не проведена» — раскладку партиций решил планировщик, и опыта не
    получилось; повторы внутри `run_until_conducted` уже исчерпаны.

    Ронять набор одинаково в обоих случаях нельзя. Красное на условиях, а не
    на предмете проверки, читается как «тест мигает», и через месяц его
    отключат — и будут правы. Пропуск с полными следами не выглядит
    пройденной проверкой и не выглядит поломкой; он ровно то, чем является.
    """
    try:
        return experiment()
    except ParallelismNotProvenError as problem:
        pytest.skip(f"условия опыта не сложились: {problem}")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_INPUT = ROOT / "data" / "golden_input"
GOLDEN_EXPECTED = ROOT / "data" / "golden_expected"
NOW = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
WORKERS = 4


@pytest.fixture(name="isolated_cwd", scope="module", autouse=True)
def isolated_cwd_fixture(tmp_path_factory):
    """Увести прогон в каталог, из которого нечего подставить.

    Утверждаемое свойство ровно одно: **ни один вход этого модуля не
    адресуется от текущего каталога**. Всё остальное — от `ROOT` или из
    временного workspace. Соседние свойства фикстура не сторожит: она
    промолчит, если вход возьмут по абсолютному пути мимо репозитория.
    """
    empty = tmp_path_factory.mktemp("isolated_cwd")
    previous = Path.cwd()
    os.chdir(empty)
    try:
        yield empty
    finally:
        os.chdir(previous)


@pytest.fixture(name="rebuilt", scope="module")
def rebuilt_fixture(tmp_path_factory, isolated_cwd) -> dict:
    """Пересобрать TRAIN и артефакты с нуля во временном каталоге.

    Каталог временный намеренно: прогон не должен трогать `data/` и
    `artifacts/` разработчика. Данные и артефакты детерминированы, поэтому
    временная копия побайтно совпадает с постоянной.

    `isolated_cwd` запрошен явно, хотя он и autouse: пересборка — первое, что
    здесь читает файлы, и порядок «сначала уйти из репозитория, потом читать»
    не должен держаться на порядке autouse-фикстур.
    """
    workspace = tmp_path_factory.mktemp("conformance")
    raw = workspace / "raw"
    artifacts = workspace / "artifacts"

    build_dataset(GeneratorConfig(out_dir=raw))

    settings = PreprocessingSettings()
    train = TrainDataset.load(raw)
    result = run_build(
        train, config_dir=ROOT / "config", settings=settings, processing_time=NOW
    )
    freeze_build(result, artifacts, build_timestamp=NOW, root=ROOT)

    return {
        "raw": raw,
        "artifacts": artifacts,
        "train": train,
        "settings": settings,
        "state_sha256": result.state_sha256,
    }


# --------------------------------------------------------------------------- #
# §29.2 — golden-vectors
# --------------------------------------------------------------------------- #


def test_golden_expected_is_reproduced_from_seed(rebuilt: dict):
    """Полная цепочка `seed` → данные → артефакты → эталон (§29.2 пп. 2, 3).

    Проверяется не «прогон совпал с прогоном», а что замороженный эталон
    воспроизводится из того, что лежит в git: конфигов, `seed` генератора и
    `golden_input`. Ничего промежуточного хранить для этого не требуется.
    """
    artifacts = FrozenArtifacts.load(rebuilt["artifacts"], "0.1.0")
    dataset = Dataset.load(GOLDEN_INPUT)

    result = run_encode(
        dataset,
        artifacts=artifacts,
        config_dir=ROOT / "config",
        settings=rebuilt["settings"],
        processing_time=NOW,
    )

    compare_expected(build_expected(result), load_expected(GOLDEN_EXPECTED))


def test_rebuilt_state_matches_the_frozen_hash(rebuilt: dict):
    """Пересобранное состояние совпало с тем, при котором снят эталон (§30).

    Это и есть ответ на вопрос, зачем не хранить артефакты в git: если хэш
    состояния сошёлся, пересборка дала те же артефакты, и промежуточное
    хранение ничего бы не добавило. Не сошёлся — расхождение видно здесь, а
    не через сломанный эталон непонятно почему.
    """
    frozen = load_expected(GOLDEN_EXPECTED)

    assert rebuilt["state_sha256"] == frozen.preprocessing_state_sha256


# --------------------------------------------------------------------------- #
# §29 п.10 — single vs multi-worker
# --------------------------------------------------------------------------- #


def test_build_does_not_depend_on_the_number_of_workers(rebuilt: dict):
    """BUILD в один процесс и в четыре даёт те же артефакты байт-в-байт.

    `compare_workers` сначала доказывает, что параллельность состоялась
    (разные PID, непустое распределение, порядок завершения отличается от
    канонического), и только потом сравнивает. Вырожденный прогон объявляется
    непроведённой проверкой, а не пройденной.
    """
    evidence = conducted(
        lambda: compare_workers(
            rebuilt["train"],
            config_dir=ROOT / "config",
            settings=rebuilt["settings"],
            processing_time=NOW,
            workers=WORKERS,
        )
    )

    assert len(evidence.by_pid) >= 2
    assert evidence.completion_order != evidence.canonical_order


def test_encode_does_not_depend_on_the_number_of_workers(rebuilt: dict):
    """`prepared_*` не зависят от числа воркеров — на golden-наборе.

    Именно на golden: он маленький, и при этом покрывает все краевые случаи
    §29.2, то есть проверка идёт по самым неудобным данным, а не по средним.
    """
    artifacts = FrozenArtifacts.load(rebuilt["artifacts"], "0.1.0")

    evidence = conducted(
        lambda: compare_workers_encode(
            Dataset.load(GOLDEN_INPUT),
            artifacts=artifacts,
            config_dir=ROOT / "config",
            settings=rebuilt["settings"],
            processing_time=NOW,
            workers=WORKERS,
        )
    )

    assert len(evidence.by_pid) >= 2
