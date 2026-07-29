"""Точка входа генератора: собирает датасеты и пишет манифесты.

Собирается два набора:

- `main` — рабочий датасет для BUILD и подбора bucket_edges;
- `golden_input` — маленький замороженный набор из §29.2: те же клиенты и те же
  краевые случаи, но минимум фонового шума. Позже именно из него первый
  эталонный прогон ENCODE даст `golden_expected`.

Запуск:
    python -m src.generator.build_dataset
    python -m src.generator.build_dataset --only golden
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import chain
from pathlib import Path
from typing import Iterator

from .clients import build_clients, identity_mapping
from .config import DatasetRole, GeneratorConfig
from .edge_cases import MIN_CLIENTS, CaseLog, CasePlan, assign_cases, build_plan
from .records import RawRecord
from .sources import (
    SOURCE_NAMES,
    app_logs,
    card_processing,
    core_payments,
    fx_rates,
    profile_snapshots,
)
from .writer import FileInfo, JsonlWriter

logger = logging.getLogger(__name__)

META_DIR = "_meta"

# Golden-набор: те же роли краевых случаев, но фоновых событий в разы меньше —
# набор должен оставаться обозримым и сравниваться байт-в-байт.
GOLDEN_VOLUME_SCALE = 0.12
GOLDEN_OUT_DIR = Path("data/golden_input")


def golden_clients(config: GeneratorConfig) -> int:
    """Размер golden-набора: по клиенту на роль плюс обещанные чистые.

    Роли раздаются первым `MIN_CLIENTS` клиентам, поэтому всё сверх них —
    клиенты без единого краевого случая. Эталон из одних краевых случаев
    неполон: happy path в нём не представлен, и поломка обычного пути прошла
    бы мимо golden-векторов.
    """
    return MIN_CLIENTS + config.debug_clean_clients


def golden_config(base: GeneratorConfig) -> GeneratorConfig:
    """Конфиг golden-набора: тот же seed и тот же период, меньше объёма."""
    return base.model_copy(
        update={
            "n_clients": golden_clients(base),
            "volume_scale": GOLDEN_VOLUME_SCALE,
            "out_dir": GOLDEN_OUT_DIR,
            "dataset_name": "golden_input",
            "dataset_role": DatasetRole.GOLDEN_INPUT,
        }
    )


def build(config: GeneratorConfig) -> dict:
    """Сгенерировать один датасет и вернуть его манифест."""
    clients = assign_cases(build_clients(config), config)
    plan = build_plan(config)
    case_log = CaseLog()

    written = JsonlWriter(config.out_dir).write(_all_records(clients, config, plan, case_log))

    missing = case_log.missing_cases()
    if missing:
        raise ValueError(
            "в датасете не оказалось краевых случаев: " + ", ".join(missing)
        )

    manifest = _manifest(config, clients, written)
    _write_meta(config, "manifest.json", manifest)
    _write_meta(config, "identity_mapping.json", identity_mapping(clients))
    _write_meta(config, "edge_case_manifest.json", case_log.as_manifest())

    logger.info(
        "%s: %d файлов, %d записей, %d краевых случаев → %s",
        config.dataset_name,
        len(written),
        manifest["total_records"],
        len(case_log.as_manifest()),
        config.out_dir,
    )
    return manifest


def _all_records(
    clients, config: GeneratorConfig, plan: CasePlan, case_log: CaseLog
) -> Iterator[RawRecord]:
    return chain(
        core_payments.generate(clients, config, plan, case_log),
        card_processing.generate(clients, config, plan, case_log),
        app_logs.generate(clients, config, plan, case_log),
        profile_snapshots.generate(clients, config, plan, case_log),
        fx_rates.generate(config, plan, case_log),
    )


def _manifest(config: GeneratorConfig, clients, written: list[FileInfo]) -> dict:
    return {
        "dataset_name": config.dataset_name,
        # Роль решает, что с набором можно делать: BUILD (§27) запускается
        # только на TRAIN. Публикуется с данными, а не выводится из имени
        # каталога — по имени это была бы догадка.
        "dataset_role": str(config.dataset_role),
        "generator_version": config.generator_version,
        "golden_vectors_version": config.golden_vectors_version,
        "seed": config.seed,
        "n_clients": len(clients),
        # Обещание набора трассировке (1.6): столько клиентов в нём заведомо
        # без краевых случаев. Читает это поле debug-режим препроцессинга —
        # он и так берёт из `_meta` две таблицы, поэтому связь идёт по данным,
        # а не импортом из кода генератора.
        "debug_clean_clients": config.debug_clean_clients,
        "volume_scale": config.volume_scale,
        "history_start": config.history_start.isoformat(),
        "history_end": config.history_end.isoformat(),
        "cutoff_time": config.cutoff_time.isoformat().replace("+00:00", "Z"),
        "sources": list(SOURCE_NAMES),
        "total_records": sum(item.records for item in written),
        "files": [
            {"path": item.path, "records": item.records, "sha256": item.sha256}
            for item in written
        ],
    }


def _write_meta(config: GeneratorConfig, name: str, payload: object) -> None:
    meta_dir = config.out_dir / META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (meta_dir / name).write_bytes(text.encode("utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генератор синтетических банковских данных")
    parser.add_argument("--seed", type=int, help="переопределить seed")
    parser.add_argument("--clients", type=int, help="число клиентов основного набора")
    parser.add_argument("--out", type=Path, help="каталог основного набора")
    parser.add_argument(
        "--only",
        choices=("main", "golden"),
        help="собрать только один набор (по умолчанию оба)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    overrides = {
        key: value
        for key, value in (
            ("seed", args.seed),
            ("n_clients", args.clients),
            ("out_dir", args.out),
        )
        if value is not None
    }
    base = GeneratorConfig(**overrides)

    if args.only != "golden":
        build(base)
    if args.only != "main":
        build(golden_config(base))


if __name__ == "__main__":
    main()
