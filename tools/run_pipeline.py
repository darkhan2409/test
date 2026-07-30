"""Точка входа препроцессинга: BUILD → заморозка → ENCODE → выдача.

Своей логики обработки здесь нет ни одной строки: скрипт вызывает публичный API
в порядке, который задают §27 и §28, и печатает итоги. Понадобился он потому,
что до сих пор запускать пайплайн было нечем — `write_prepared`,
`build_run_report` и `Quarantine.write` определены, но не вызывались нигде.

Запуск:
    python -m tools.run_pipeline            # обычный прогон
    python -m tools.run_pipeline --debug    # плюс трассировка §1.6 в data/debug/
    python -m tools.run_pipeline --debug --skip-build   # артефакты уже собраны
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.core.quarantine import Quarantine
from src.preprocessing.core.run_report import build_run_report, write_run_report
from src.preprocessing.core.settings import PreprocessingSettings
from src.preprocessing.pipeline import (
    PIPELINE_VERSION,
    Dataset,
    FrozenArtifacts,
    TrainDataset,
    freeze_build,
    run_build,
    run_encode,
)
from src.preprocessing.prepared_output import write_prepared

UTC = timezone.utc
logger = logging.getLogger("run_pipeline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогон препроцессинга: BUILD и ENCODE")
    parser.add_argument("--debug", action="store_true", help="писать трассировку в data/debug/")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="не пересобирать артефакты, взять уже замороженные",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    root = Path(__file__).resolve().parent.parent
    settings = PreprocessingSettings(debug=args.debug)
    config_dir = root / settings.config_dir
    artifacts_dir = root / settings.artifacts_dir
    raw_dir = root / settings.raw_dir
    now = datetime.now(UTC)

    if not args.skip_build:
        started = time.monotonic()
        # §27: BUILD выполняется только на TRAIN — роль читается из манифеста.
        train = TrainDataset.load(raw_dir)
        build = run_build(
            train, config_dir=config_dir, settings=settings, processing_time=now
        )
        freeze_build(build, artifacts_dir, build_timestamp=now, root=root)
        logger.info(
            "BUILD: набор %s, состояние %s, %.1f c",
            train.identifier,
            build.state_sha256[:12],
            time.monotonic() - started,
        )

    started = time.monotonic()
    artifacts = FrozenArtifacts.load(artifacts_dir, PIPELINE_VERSION)
    dataset = Dataset.load(raw_dir)

    # Монитор и карантин создаются здесь, а не внутри: карантин нужно записать
    # на диск после прогона, и §34 требует, чтобы он поднимал метрики в тот же
    # монитор, который попадёт в отчёт.
    monitor = DataQualityMonitor()
    quarantine = Quarantine(
        monitor, processing_time=now, pipeline_version=PIPELINE_VERSION
    )

    result = run_encode(
        dataset,
        artifacts=artifacts,
        config_dir=config_dir,
        settings=settings,
        processing_time=now,
        monitor=monitor,
        quarantine=quarantine,
    )

    prepared_dir = root / settings.prepared_dir
    contract = write_prepared(result, prepared_dir / dataset.identifier)
    write_run_report(
        prepared_dir,
        build_run_report(
            monitor=monitor,
            quarantine=quarantine,
            run_started_at=now,
            versions=dict(result.metadata),
        ),
    )
    quarantine.write(root / settings.quarantine_dir)

    logger.info(
        "ENCODE: %d событий, %d профилей, карантин %d, %.1f c",
        len(result.events()),
        len(result.profiles()),
        result.quarantine["total"],
        time.monotonic() - started,
    )
    logger.info("выход: %s (%d пунктов контракта §2)", prepared_dir / dataset.identifier,
                len(contract.manifest()))

    if args.debug:
        debug_dir = root / settings.debug_dir
        components = sorted(item.name for item in debug_dir.iterdir() if item.is_dir())
        logger.info("трассировка: %s, каталогов %d", debug_dir, len(components))


if __name__ == "__main__":
    main()
