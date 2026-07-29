"""Обязательные артефакты — §31.

§31 перечисляет тридцать два пункта, которые обязаны пережить прогон. Список
разнородный: конфиги, посчитанные границы, политики, идентификаторы и
происхождение кода. Общее у них одно — по ним восстанавливают, чем именно
получены данные, и пункт, которого нет, обнаруживается через месяцы, когда
восстанавливать уже поздно.

**Полнота — сигнатурой**, как в §30 и §2. `RequiredArtifacts` — frozen
dataclass с тридцатью двумя обязательными полями; перечень §31 продублирован
константой и сверяется с полями при импорте.

**Что в хэш не входит и почему.** Три пункта §31 описывают прогон, а не
преобразование: `build timestamp`, `TRAIN baselines` и происхождение кода.
Время меняется каждый запуск — попади оно в `preprocessing_state_sha256`, два
одинаковых BUILD дали бы разные хэши, и сравнение single/multi-worker (§29
п.10) развалилось бы на пустом месте. Baselines только сравнивают при
мониторинге. Критерий один и тот же: **участвует ли артефакт в
преобразовании данных**. `bucket_edges` участвуют — они в хэше, хотя тоже
посчитаны из данных.

**Происхождение кода: хэш содержимого, а не имя коммита.** `git commit`
описывает код честно только при чистом рабочем дереве. BUILD, запущенный с
незакоммиченными правками, записал бы «собрано коммитом X», а пересборка из
X дала бы другое — тихая ложь, которая вскрывается позже всего. Поэтому
основной пункт — `code_state_sha256`, хэш фактического содержимого
`src/`, `tools/` и `config/`: он описывает код, который действительно
отработал, и соврать не может по построению. Коммит и флаг `dirty` пишутся
рядом справкой.

Блокировать грязное дерево здесь не нужно: отладочный прогон идёт с правками
постоянно, и запрет заставил бы коммитить ради запуска. Цена лжи разная у
разных артефактов, поэтому запрет стоит там, где она высока — заморозка
`golden_expected` (§29.2 п.3, «несовпадение — блокирующая ошибка релиза»)
требует чистого дерева, см. `require_clean_tree`.
"""

from __future__ import annotations

import subprocess
from dataclasses import MISSING as NO_DEFAULT
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.canonical import canonical_bytes
from .core.hashing import HASH_POLICY, content_hex
from .field_policies import TEXT_POLICY
from .sampler import SAMPLING_ALGORITHM

COMPONENT = "artifacts"

# Каталоги, содержимое которых и есть «код прогона». `tests/` сюда не входит:
# тесты не участвуют в обработке, и их правка не меняет результат.
CODE_ROOTS: tuple[str, ...] = ("src", "tools", "config")
CODE_SUFFIXES: frozenset[str] = frozenset({".py", ".yaml", ".yml"})

# Перечень §31, дословно и в порядке регламента. Имена приведены к snake_case
# там, где регламент пишет их словами («TRAIN baselines», «code commit/hash»).
SPEC_31_ARTIFACTS: tuple[str, ...] = (
    "source_contracts",
    "identity_mapping",
    "event_mapping",
    "feature_schema",
    "closed_set_domains",
    "bucket_field_domains",
    "category_mappings",
    "dedup_config",
    "sessionization_config",
    "timestamp_policy",
    "calendar_timezone_policy",
    "cutoff_policy",
    "fx_normalization_config",
    "fx_max_staleness",
    "bucket_edges",
    "bucket_metadata",
    "time_delta_edges",
    "numeric_validation_rules",
    "high_cardinality_policy",
    "text_policy",
    "field_priorities",
    "max_values_per_field",
    "train_baselines",
    "deterministic_build_config",
    "hash_policy",
    "golden_vectors",
    "sampling_algorithm",
    "global_seed",
    "preprocessing_state_sha256",
    "code_commit_hash",
    "train_dataset_identifier",
    "build_timestamp",
)


class ArtifactsError(RuntimeError):
    """Артефакты собрать нельзя — блокирующая ошибка."""


@dataclass(frozen=True)
class RequiredArtifacts:
    """Тридцать два пункта §31 — по одному полю на пункт, без умолчаний."""

    source_contracts: Any
    identity_mapping: Any
    event_mapping: Any
    feature_schema: Any
    closed_set_domains: Any
    bucket_field_domains: Any
    category_mappings: Any
    dedup_config: Any
    sessionization_config: Any
    timestamp_policy: Any
    calendar_timezone_policy: Any
    cutoff_policy: Any
    fx_normalization_config: Any
    fx_max_staleness: Any
    bucket_edges: Any
    bucket_metadata: Any
    time_delta_edges: Any
    numeric_validation_rules: Any
    high_cardinality_policy: Any
    text_policy: Any
    field_priorities: Any
    max_values_per_field: Any
    train_baselines: Any
    deterministic_build_config: Any
    hash_policy: Any
    golden_vectors: Any
    sampling_algorithm: Any
    global_seed: Any
    preprocessing_state_sha256: Any
    code_commit_hash: Any
    train_dataset_identifier: Any
    build_timestamp: Any

    def document(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in dataclass_fields(self)}


def code_provenance(root: Path) -> dict[str, Any]:
    """Происхождение кода — §31 «code commit/hash».

    `code_state_sha256` считается по каноническому списку файлов: путь в
    POSIX-виде плюс хэш содержимого, отсортированные по пути. Файловая
    система порядок не решает (§29 п.1), и результат один на любой машине.
    """
    entries: list[dict[str, str]] = []
    for name in CODE_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in CODE_SUFFIXES:
                continue
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": content_hex(path.read_bytes()),
                }
            )

    if not entries:
        raise ArtifactsError(f"в {root} нет ни одного файла кода — нечего описывать")

    entries.sort(key=lambda item: item["path"])
    commit, dirty = _git_state(root)
    return {
        # Основной пункт: описывает код, который отработал, а не имя, под
        # которым он, возможно, лежит в истории.
        "code_state_sha256": content_hex(canonical_bytes(entries)),
        "files": len(entries),
        "roots": list(CODE_ROOTS),
        # Справка. При грязном дереве коммит описывает не тот код, поэтому
        # он идёт вместе с флагом и никогда вместо `code_state_sha256`.
        "git_commit": commit,
        "dirty": dirty,
    }


def _git_state(root: Path) -> tuple[str | None, bool]:
    """Коммит и признак незакоммиченных правок.

    Отсутствие git — не ошибка: код мог приехать архивом, и тогда честный
    ответ «коммита нет», а не выдуманный ноль.
    """
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return None, False
    status = run("status", "--porcelain")
    return commit, bool(status)


def require_clean_tree(provenance: Mapping[str, Any], purpose: str) -> None:
    """Запретить действие, если код собран из грязного дерева.

    Ставится не на каждый BUILD: отладочный прогон идёт с правками
    постоянно, и запрет заставил бы коммитить ради запуска. Ставится там, где
    ложь дорога — заморозка эталона (§29.2 п.3).
    """
    if provenance.get("dirty"):
        raise ArtifactsError(
            f"{purpose}: рабочее дерево грязное, и записанный git_commit "
            f"{provenance.get('git_commit')} описывает не тот код, который отработал. "
            "Закоммитьте правки — эталон снимается с воспроизводимого состояния (§29.2)"
        )


def deterministic_build_config(settings: Any) -> dict[str, Any]:
    """`deterministic_build_config` — §31.

    Собирает в одном месте всё, от чего зависит воспроизводимость BUILD:
    что сортируется, чем задаётся выборка и что в хэш не входит. Значения
    берутся из кода и настроек, а не переписываются руками — иначе документ
    разойдётся с поведением при первой же правке.
    """
    from .core.canonical import SERIALIZATION_CONFIG
    from .pipeline import PartitionOrder

    return {
        "partition_order": str(PartitionOrder.CANONICAL),
        "partition_sort_key": "canonical_path (§29 п.1)",
        "merge_order": "строгий порядок партиций (§29 п.3)",
        "global_seed": settings.global_seed,
        "bucket_sample_size": settings.bucket_sample_size,
        "sampling_algorithm": SAMPLING_ALGORITHM["name"],
        "serialization_config_version": SERIALIZATION_CONFIG["serialization_config_version"],
        "hash_policy_version": HASH_POLICY["hash_policy_version"],
        # Прямая запись того, что в хэш состояния НЕ входит: список короткий,
        # а вопрос «почему хэш не изменился» задают регулярно.
        "excluded_from_state_hash": [
            "build_timestamp — меняется каждый прогон, в преобразовании не участвует",
            "train_baselines — сравниваются при мониторинге, ни одного значения не меняют",
            "code_state_sha256 — описывает прогон, а не конфигурацию преобразования",
            "окружение: пути, число воркеров, debug (§29 п.10)",
        ],
    }


def collect_artifacts(
    result: Any,
    *,
    root: Path,
    build_timestamp: datetime,
    golden_input_dir: Path,
    golden_expected_dir: Path | None = None,
) -> RequiredArtifacts:
    """Собрать перечень §31 из результата BUILD.

    `golden_expected` на этом шаге ещё не существует: §29.2 п.1 требует
    заполнить его первым эталонным прогоном ENCODE, то есть на 4.2. Пункт от
    этого не исчезает — он честно сообщает, что заморожен только вход.
    """
    configs = result.configs
    schema = configs.schema
    edges = result.bucket_edges
    settings_policy = result.state.document()["run_policy"]

    from .cutoff import cutoff_policy_state

    return RequiredArtifacts(
        source_contracts=configs.registry.state(),
        identity_mapping=configs.identity.state(),
        event_mapping=configs.events.state(),
        feature_schema=schema.state(),
        closed_set_domains={
            name: list(values) for name, values in schema.closed_set_domains().items()
        },
        bucket_field_domains={
            name: list(values) for name, values in edges.bucket_field_domains().items()
        },
        category_mappings=configs.categories.state(),
        dedup_config=configs.dedup.state(),
        sessionization_config=configs.sessionization.state(),
        timestamp_policy=configs.timestamps.state(),
        calendar_timezone_policy={
            "calendar_timezone_policy_version": configs.timestamps.calendar_timezone_policy_version,
            "timezone_mappings": {
                name: dict(sorted(values.items()))
                for name, values in sorted(configs.timestamps.timezone_mappings.items())
            },
        },
        cutoff_policy=cutoff_policy_state(result.cutoff_time),
        fx_normalization_config=configs.fx.state(),
        fx_max_staleness=settings_policy["fx_max_staleness_days"],
        bucket_edges=edges.state(),
        bucket_metadata=edges.bucket_metadata(),
        time_delta_edges=result.time_delta_edges.state(),
        numeric_validation_rules=schema.numeric_rules(),
        high_cardinality_policy={
            name: str(spec.high_cardinality)
            for name, spec in schema.field_specs().items()
            if spec.high_cardinality is not None
        },
        text_policy=dict(TEXT_POLICY),
        field_priorities=schema.field_priority(),
        max_values_per_field=schema.max_values_per_field(),
        train_baselines=dict(result.baselines),
        deterministic_build_config=deterministic_build_config(result.settings),
        hash_policy=dict(HASH_POLICY),
        golden_vectors={
            "golden_vectors_version": result.versions.golden_vectors_version,
            "golden_input": golden_input_dir.as_posix(),
            "golden_expected": (
                golden_expected_dir.as_posix() if golden_expected_dir else None
            ),
            "expected_frozen": golden_expected_dir is not None,
            "note": (
                "golden_expected заполняется первым эталонным прогоном ENCODE (§29.2 п.1) "
                "и не раньше зелёного сравнения single/multi-worker"
            ),
        },
        sampling_algorithm=dict(SAMPLING_ALGORITHM),
        global_seed=settings_policy["global_seed"],
        preprocessing_state_sha256=result.state_sha256,
        code_commit_hash=code_provenance(root),
        train_dataset_identifier=result.dataset.identifier,
        # Время прогона. В состояние §30 оно не входит и войти не может:
        # `build_state` собирается из конфигов и границ, времени там нет.
        build_timestamp=build_timestamp.isoformat().replace("+00:00", "Z"),
    )


def write_artifacts(artifacts: RequiredArtifacts, target: Path) -> list[Path]:
    """Записать каждый пункт §31 отдельным каноническим файлом.

    Один пункт — один файл: перечень §31 читается как листинг каталога, и
    «чего-то нет» видно без чтения кода.
    """
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in sorted(artifacts.document().items()):
        path = target / f"{name}.json"
        path.write_bytes(canonical_bytes(payload))
        written.append(path)
    return written


def _check_artifacts_are_complete() -> None:
    """Перечень §31 совпадает с полями, и ни у одного нет умолчания.

    Та же пара проверок, что у состояния §30 и контракта §2, и по той же
    причине: умолчание вернуло бы возможность не заполнить пункт.
    """
    declared = {item.name for item in dataclass_fields(RequiredArtifacts)}

    missing = sorted(set(SPEC_31_ARTIFACTS) - declared)
    if missing:
        raise RuntimeError("в артефактах нет пунктов §31: " + ", ".join(missing))

    extra = sorted(declared - set(SPEC_31_ARTIFACTS))
    if extra:
        raise RuntimeError("в артефактах есть пункты сверх §31: " + ", ".join(extra))

    optional = sorted(
        item.name
        for item in dataclass_fields(RequiredArtifacts)
        if item.default is not NO_DEFAULT or item.default_factory is not NO_DEFAULT
    )
    if optional:
        raise RuntimeError(
            "пункты §31 со значением по умолчанию: "
            + ", ".join(optional)
            + " — такой пункт можно не заполнить, и артефакты уедут неполными"
        )


_check_artifacts_are_complete()
