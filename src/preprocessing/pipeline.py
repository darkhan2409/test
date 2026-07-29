"""BUILD PHASE — §27, §37.1.

BUILD считает то, что потом замораживается: границы бакетов, границы дельты,
domain бакет-полей, baselines и `preprocessing_state_sha256`. Всё остальное
(разметка событий, выдача `prepared_*`) — работа ENCODE, §28.

**«BUILD не выполняется на отдельном примере, batch, Validation, Test или
inference»** (§27). Запрет выражен двумя способами, и оба структурные.
Во-первых, у `run_build` нет параметра, которым можно передать пример или
батч: он принимает `TrainDataset` — целый набор на диске. Во-вторых,
`TrainDataset` собирается только из набора, чей манифест объявил роль `train`;
проверка стоит один раз на границе, а дальше роль гарантирует тип. Позвать
BUILD на golden-наборе не получится не потому, что кто-то это заметит, а
потому, что объект-аргумент из него не построится.

**Шаг 10 «отобрать TRAIN без leakage» на этих данных вырожден,** и это стоит
сказать прямо. Разбиений Validation/Test генератор не делает, поэтому TRAIN —
весь набор целиком; от утечки будущего защищает отсечка §14, которая стоит в
цепочке раньше и уже отработала. Когда разбиения появятся, отбор станет
отдельным шагом с собственной проверкой, а сейчас изображать его нечем.

**Шаг 16 «single/multi-worker equality» здесь выполняется частично.** Проверка
идёт по свойству, которое требует §29 пп. 5–6: результат не зависит от порядка,
в котором обработаны и слиты партиции. BUILD прогоняется дважды — партиции в
каноническом порядке и в обратном, — и артефакты сравниваются побайтно.
Сравнение процессов (1 воркер против 4, `spawn`) остаётся за шагом 4.1 плана:
там оно охватывает и ENCODE, а здесь дало бы половину проверки под видом
целой.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_hasher import PreprocessingState, build_state
from .bucketizer import BucketEdges, BucketizationConfig, fit_bucket_edges, load_bucketization_config
from .category_normalizer import CategoryMapping, CategoryNormalizer, load_category_mapping
from .core.canonical import canonical_bytes, canonical_text
from .core.monitor import DataQualityMonitor
from .core.quarantine import Quarantine
from .core.settings import PreprocessingSettings
from .core.versions import PreprocessingVersions
from .cutoff import CutoffFilter
from .deduplicator import DedupPolicy, Deduplicator, load_dedup_policy
from .event_mapper import EventMapper, EventMapping, load_event_mapping
from .feature_projection import (
    FeatureProjector,
    MissingPolicy,
    ProjectedRecord,
    load_feature_schema,
)
from .field_policies import TEXT_POLICY_VERSION, FieldPolicies, check_field_policies
from .fx_normalizer import FxConfig, FXNormalizer, FxRateTable, load_fx_config
from .identity_resolver import IdentityMapping, IdentityResolver, load_identity_mapping
from .numeric_validator import NumericValidator
from .profile_builder import ProfileBuilder, ProfilePolicy, load_profile_policy
from .sampler import DeterministicSampler
from .schema import load_source_contracts
from .schema.feature_schema import FeatureSchema
from .schema.source_contract import SourceContractRegistry
from .sessionizer import SessionizationConfig, Sessionizer, load_sessionization_config
from .source_reader import SourceReader
from .time_delta import (
    DeltaSampler,
    TimeDeltaConfig,
    TimeDeltaEdges,
    collect_deltas,
    fit_time_delta_edges,
    load_time_delta_config,
)
from .time_feature_builder import TimeFeatureBuilder
from .timeline_builder import TimelineBuilder
from .timestamp_normalizer import (
    ClientTimezoneIndex,
    TimestampNormalizer,
    TimestampPolicy,
    load_timestamp_policy,
)

DATASET_MANIFEST = "_meta/manifest.json"
TRAIN_ROLE = "train"

PIPELINE_VERSION = "0.1.0"
"""`preprocessing_version` (§30). Объявляется здесь, а не берётся из конфига:
это версия самого кода пайплайна, и менять её должен тот, кто меняет код."""


class PartitionOrder(StrEnum):
    """Порядок обхода партиций.

    `CANONICAL` — единственный, в котором BUILD запускается по-настоящему
    (§29 пп. 1–2). `REVERSED` существует ровно для шага 16: артефакты,
    посчитанные в обратном порядке, обязаны совпасть с каноническими побайтно.
    Третьего значения нет — «случайный порядок» сделал бы проверку
    невоспроизводимой, а невоспроизводимая проверка хуже отсутствующей.
    """

    CANONICAL = "canonical"
    REVERSED = "reversed"

# Артефакты BUILD. Имена — те же, что в перечнях §30 и §31, чтобы каталог
# читался рядом с регламентом, а не расшифровывался.
STATE_FILE = "preprocessing_state.json"
VERSIONS_FILE = "versions.json"
BUCKET_EDGES_FILE = "bucket_edges.json"
TIME_DELTA_EDGES_FILE = "time_delta_edges.json"
BUCKET_FIELD_DOMAINS_FILE = "bucket_field_domains.json"
BUCKET_METADATA_FILE = "bucket_metadata.json"
BASELINES_FILE = "train_baselines.json"


class BuildPhaseError(RuntimeError):
    """Ошибка BUILD — блокирующая. Замораживать нечего, пока она не устранена."""


# --------------------------------------------------------------------------- #
# Шаг 10: на чём BUILD вообще разрешён
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainDataset:
    """Набор, на котором BUILD разрешён (§27).

    Существование объекта — уже утверждение, что роль набора `train`.
    Собирается только через `load`, где роль читается из манифеста.
    """

    identifier: str
    raw_dir: Path
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, raw_dir: Path) -> "TrainDataset":
        """Прочитать манифест набора и убедиться, что это TRAIN.

        Роль берётся из данных, а не из имени каталога: каталог можно
        переименовать, скопировать и смонтировать куда угодно, а манифест
        едет вместе с записями.
        """
        manifest_path = raw_dir / DATASET_MANIFEST
        if not manifest_path.exists():
            raise BuildPhaseError(
                f"нет манифеста набора {manifest_path}: неизвестно, TRAIN ли это (§27)"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        role = manifest.get("dataset_role")
        if role != TRAIN_ROLE:
            raise BuildPhaseError(
                f"{raw_dir}: роль набора {role!r}, а BUILD выполняется только на "
                f"{TRAIN_ROLE!r} (§27)"
            )

        identifier = manifest.get("dataset_name")
        if not identifier:
            # §31 требует хранить идентификатор TRAIN-датасета. Артефакт без
            # него не сказал бы, на чём посчитан.
            raise BuildPhaseError(f"{manifest_path}: в манифесте нет dataset_name (§31)")

        return cls(identifier=identifier, raw_dir=raw_dir, manifest=manifest)


# --------------------------------------------------------------------------- #
# Шаги 1–9: заморозка конфигов
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrozenConfigs:
    """Конфигурация пайплайна, загруженная и проверенная (§27 шаги 1–9).

    «Заморозка» здесь не операция, а свойство: все модели `frozen=True`, а
    взаимные проверки (схема против контрактов, сессионизация против маппинга
    событий, алиасы против domain) уже выполнены загрузчиками. Поменять что-то
    после этой точки можно только собрав набор заново.
    """

    registry: SourceContractRegistry
    identity: IdentityMapping
    timestamps: TimestampPolicy
    dedup: DedupPolicy
    events: EventMapping
    sessionization: SessionizationConfig
    schema: FeatureSchema
    missing: MissingPolicy
    categories: CategoryMapping
    fx: FxConfig
    bucketization: BucketizationConfig
    time_delta: TimeDeltaConfig
    profile: ProfilePolicy

    def versions(self) -> PreprocessingVersions:
        """Комплект версий §30, собранный из самих конфигов.

        Собирается здесь, а не перечисляется руками в артефакте: версия и
        конфиг, из которого она взята, обязаны приезжать вместе. `UNSET`
        остаётся только у того, чего в конфигах нет по существу, — эти поля
        дописывает `run_build`.
        """
        return PreprocessingVersions(
            source_contract_version=self.registry.source_contract_version,
            identity_mapping_version=self.identity.version,
            event_mapping_version=self.events.event_mapping_version,
            feature_schema_version=self.schema.version,
            # §30 перечисляет domain-версии отдельно, но собственных
            # версионируемых файлов у них нет: closed-set domains объявлены
            # внутри Feature Schema, bucket-domains публикует §19 по границам.
            # Поэтому версия наследуется от источника истины, а не выдумывается.
            closed_set_domains_version=self.schema.version,
            bucket_field_domains_version=self.bucketization.bucket_edges_version,
            category_mapping_version=self.categories.category_mapping_version,
            timestamp_policy_version=self.timestamps.timestamp_policy_version,
            calendar_timezone_policy_version=self.timestamps.calendar_timezone_policy_version,
            dedup_policy_version=self.dedup.dedup_policy_version,
            sessionization_version=self.sessionization.sessionization_version,
            fx_normalization_version=self.fx.fx_normalization_version,
            bucket_edges_version=self.bucketization.bucket_edges_version,
            time_delta_edges_version=self.time_delta.time_delta_edges_version,
            text_policy_version=TEXT_POLICY_VERSION,
        )


def freeze_configs(config_dir: Path, settings: PreprocessingSettings) -> FrozenConfigs:
    """§27 шаги 1–9: загрузить и проверить всё, что BUILD замораживает."""
    registry = load_source_contracts(config_dir / "source_contracts.yaml")
    identity = load_identity_mapping(config_dir / "identity_mapping.yaml", registry)
    timestamps = load_timestamp_policy(config_dir / "timestamp_policy.yaml", registry)
    dedup = load_dedup_policy(config_dir / "dedup_policy.yaml", registry)
    events = load_event_mapping(config_dir / "event_mapping.yaml", registry)
    sessionization = load_sessionization_config(
        config_dir / "sessionization.yaml", registry, events
    )
    schema, missing = load_feature_schema(
        config_dir / "feature_schema.yaml", registry, events.event_types
    )
    categories = load_category_mapping(config_dir / "category_mapping.yaml", schema)
    fx = load_fx_config(config_dir / "fx_config.yaml", registry)
    bucketization = load_bucketization_config(config_dir / "bucketization.yaml", schema)
    time_delta = load_time_delta_config(config_dir / "time_delta.yaml")
    profile = load_profile_policy(
        config_dir / "profile_policy.yaml", registry, schema, events.event_types
    )

    # §21–§23 проверяются на схеме и контрактах, до единой записи: ошибка в
    # политике полей — ошибка конфигурации, и ловить её на середине прогона
    # незачем.
    check_field_policies(schema, registry, default_max_values=settings.max_values_per_field)

    return FrozenConfigs(
        registry=registry,
        identity=identity,
        timestamps=timestamps,
        dedup=dedup,
        events=events,
        sessionization=sessionization,
        schema=schema,
        missing=missing,
        categories=categories,
        fx=fx,
        bucketization=bucketization,
        time_delta=time_delta,
        profile=profile,
    )


# --------------------------------------------------------------------------- #
# Цепочка §37.2 до бакетизации
# --------------------------------------------------------------------------- #


def prepare_records(
    configs: FrozenConfigs,
    settings: PreprocessingSettings,
    raw_dir: Path,
    *,
    monitor: DataQualityMonitor,
    quarantine: Quarantine,
    order: PartitionOrder = PartitionOrder.CANONICAL,
) -> list[ProjectedRecord]:
    """Цепочка §37.2 от чтения до политик полей — общая часть BUILD и ENCODE.

    Останавливается **перед** бакетизацией: на BUILD границ ещё нет, и это
    единственная точка, где два прохода расходятся. Здесь же видно, почему
    §19 не может стоять раньше: до этого места значения проходят §17 и §18 и
    только потом становятся сравнимыми числами.

    `order` задаёт порядок обхода партиций: по умолчанию канонический
    (§29 п.1), обратный нужен шагу 16, чтобы проверить независимость от него.
    """
    cutoff = settings.cutoff_time

    reader = SourceReader(configs.registry, monitor=monitor, quarantine=quarantine)
    partitions = reader.discover_partitions(raw_dir)
    if order is PartitionOrder.REVERSED:
        partitions = list(reversed(partitions))

    resolver = IdentityResolver(
        configs.registry, configs.identity, monitor=monitor, quarantine=quarantine
    )
    identified = list(
        resolver.resolve(record for item in partitions for record in reader.read(item))
    )

    index = ClientTimezoneIndex.build(
        identified, registry=configs.registry, policy=configs.timestamps, cutoff=cutoff
    )
    normalizer = TimestampNormalizer(
        configs.registry,
        configs.timestamps,
        cutoff=cutoff,
        monitor=monitor,
        quarantine=quarantine,
        client_zones=index,
    )
    cut = CutoffFilter(cutoff=cutoff, monitor=monitor)
    dedup = Deduplicator(
        configs.registry, configs.dedup, monitor=monitor, quarantine=quarantine
    )
    timed = list(dedup.deduplicate(cut.apply(normalizer.normalize(identified))))

    mapper = EventMapper(
        configs.registry, configs.events, monitor=monitor, quarantine=quarantine
    )
    sessions = Sessionizer(
        configs.sessionization,
        session_gap=settings.session_gap,
        max_values_per_field=settings.max_values_per_field,
        monitor=monitor,
    )
    projector = FeatureProjector(
        configs.schema, configs.missing, configs.registry, monitor=monitor
    )
    categories = CategoryNormalizer(
        configs.schema, configs.categories, monitor=monitor, quarantine=quarantine
    )
    numeric = NumericValidator(configs.schema, monitor=monitor)
    rates = FxRateTable.build(timed, configs.fx)
    fx = FXNormalizer(
        configs.schema,
        configs.fx,
        rates,
        max_staleness=settings.fx_max_staleness,
        monitor=monitor,
    )
    profiles = ProfileBuilder(configs.schema, configs.profile, cutoff=cutoff, monitor=monitor)
    policies = FieldPolicies(
        configs.schema, default_max_values=settings.max_values_per_field, monitor=monitor
    )

    return list(
        policies.apply(
            profiles.build(
                fx.normalize(
                    numeric.validate(
                        categories.normalize(
                            projector.project(sessions.sessionize(mapper.map(timed)))
                        )
                    )
                )
            )
        )
    )


def order_timeline(
    configs: FrozenConfigs, settings: PreprocessingSettings, records: Sequence[ProjectedRecord]
) -> list[ProjectedRecord]:
    """§13 плюс §25: порядок событий и локальные календарные признаки.

    BUILD нужен порядок, а не бакеты: дельта между соседними событиями
    считается по времени, и границы §19 на неё не влияют.
    """
    timeline = TimelineBuilder(configs.registry, cutoff=settings.cutoff_time).build(records)
    return list(TimeFeatureBuilder().build(timeline))


# --------------------------------------------------------------------------- #
# Результат BUILD
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BuildResult:
    """Всё, что BUILD посчитал и заморозил."""

    dataset: TrainDataset
    configs: FrozenConfigs
    bucket_edges: BucketEdges
    time_delta_edges: TimeDeltaEdges
    baselines: Mapping[str, Any]
    state: PreprocessingState
    versions: PreprocessingVersions

    @property
    def state_sha256(self) -> str:
        return self.state.sha256()

    def artifacts(self) -> dict[str, Any]:
        """Файлы артефактов: имя → содержимое. Пишутся канонически (§29.1)."""
        return {
            STATE_FILE: self.state.document(),
            VERSIONS_FILE: self.versions.as_metadata(),
            BUCKET_EDGES_FILE: self.bucket_edges.state(),
            TIME_DELTA_EDGES_FILE: self.time_delta_edges.state(),
            BUCKET_FIELD_DOMAINS_FILE: {
                name: list(values)
                for name, values in self.bucket_edges.bucket_field_domains().items()
            },
            BUCKET_METADATA_FILE: self.bucket_edges.bucket_metadata(),
            BASELINES_FILE: dict(self.baselines),
        }

    def fingerprint(self) -> bytes:
        """Канонический слепок всех артефактов — то, что сравнивает шаг 16."""
        return canonical_bytes(self.artifacts())


# --------------------------------------------------------------------------- #
# BUILD PHASE
# --------------------------------------------------------------------------- #


def run_build(
    dataset: TrainDataset,
    *,
    config_dir: Path,
    settings: PreprocessingSettings,
    processing_time: datetime,
    order: PartitionOrder = PartitionOrder.CANONICAL,
) -> BuildResult:
    """§27, шаги 1–18. Заморозку на диск делает `freeze_build`.

    Единственный вход — целый TRAIN-набор. Ни примера, ни батча передать
    нечем, и это не забывчивость сигнатуры (§27, последний абзац).
    """
    configs = freeze_configs(config_dir, settings)

    monitor = DataQualityMonitor()
    quarantine = Quarantine(
        monitor, processing_time=processing_time, pipeline_version=PIPELINE_VERSION
    )

    # Шаг 10. TRAIN — весь набор: разбиений нет, а от утечки будущего
    # защищает отсечка §14 внутри цепочки.
    prepared = prepare_records(
        configs, settings, dataset.raw_dir,
        monitor=monitor, quarantine=quarantine, order=order,
    )

    # Шаг 11. Детерминированная выборка.
    sampler = DeterministicSampler(
        configs.schema,
        sample_size=settings.bucket_sample_size,
        global_seed=settings.global_seed,
    )
    for record in prepared:
        sampler.offer(record)

    # Шаг 12. Границы бакетов числовых полей.
    bucket_edges = fit_bucket_edges(sampler.sample(), configs.bucketization, configs.schema)

    # Шаг 13. Границы delta-канала — отдельный компонент (§25.2).
    timeline = order_timeline(configs, settings, prepared)
    delta_sampler = DeltaSampler(
        sample_size=configs.time_delta.sample_size, global_seed=settings.global_seed
    )
    collect_deltas(timeline, delta_sampler)
    time_delta_edges = fit_time_delta_edges(delta_sampler, configs.time_delta)

    # Шаг 14. Публикация domain бакет-полей: схема получает то, чего в
    # конфиге объявить нельзя (§11.1).
    published = configs.schema.resolve_bucket_domains(bucket_edges.bucket_field_domains())
    resolved = _with_schema(configs, published)

    # Шаг 15. TRAIN baselines.
    baselines = {
        "train_dataset_identifier": dataset.identifier,
        "records": len(prepared),
        "quarantine": quarantine.summary(),
        "sampling": sampler.summary(),
        "time_delta_sampling": {
            "deltas_seen": delta_sampler.seen,
            "sample_size": len(delta_sampler.values()),
        },
        "metrics": monitor.report(),
    }

    # Шаги 17–18. Каноническая сериализация состояния и его хэш.
    state = build_state(
        source_contracts=resolved.registry,
        identity_mapping=resolved.identity,
        event_mapping=resolved.events,
        category_mapping=resolved.categories,
        feature_schema=resolved.schema,
        timestamp_policy=resolved.timestamps,
        dedup_policy=resolved.dedup,
        sessionization=resolved.sessionization,
        fx_config=resolved.fx,
        profile_policy=resolved.profile,
        bucketization=resolved.bucketization,
        bucket_edges=bucket_edges,
        time_delta_edges=time_delta_edges,
        settings=settings,
    )

    versions = resolved.versions().model_copy(
        update={
            "preprocessing_version": PIPELINE_VERSION,
            "golden_vectors_version": _golden_vectors_version(dataset),
            "fx_max_staleness_days": settings.fx_max_staleness_days,
            "preprocessing_state_sha256": state.sha256(),
        }
    )
    # §30: комплект обязан быть полным до заморозки. Пустая версия уехала бы в
    # метаданные выхода (§32.3) и всплыла бы уже на стыке с токенайзером.
    versions.require_complete()

    return BuildResult(
        dataset=dataset,
        configs=resolved,
        bucket_edges=bucket_edges,
        time_delta_edges=time_delta_edges,
        baselines=baselines,
        state=state,
        versions=versions,
    )


def check_order_independence(
    dataset: TrainDataset,
    *,
    config_dir: Path,
    settings: PreprocessingSettings,
    processing_time: datetime,
    reference: BuildResult,
) -> None:
    """§27 шаг 16, в той части, которую можно проверить здесь.

    §29 пп. 5–6 требуют, чтобы выборка и границы не зависели от порядка
    обработки и завершения задач. Проверяется это прогоном по партициям в
    обратном порядке: если где-то результат накапливается «в порядке
    поступления», обратный обход это и покажет.

    Чего проверка **не** делает: она в одном процессе, поэтому не ловит
    зависимость от рандомизации хэшей между процессами и от планировщика.
    Сравнение 1 воркер против 4 — шаг 4.1 плана.
    """
    other = run_build(
        dataset,
        config_dir=config_dir,
        settings=settings,
        processing_time=processing_time,
        order=PartitionOrder.REVERSED,
    )

    if other.fingerprint() != reference.fingerprint():
        raise BuildPhaseError(
            "артефакты BUILD зависят от порядка обхода партиций: "
            f"{other.state_sha256} против {reference.state_sha256} (§29 пп. 1–6)"
        )


def freeze_build(result: BuildResult, artifacts_dir: Path) -> list[Path]:
    """§27 шаг 19: записать артефакты и зарегистрировать версию.

    Рядом с каноническим JSON кладётся читаемая копия (допущение A5 плана).
    В хэш она не входит и входить не может: хэш считается по состоянию, а не
    по файлам каталога.
    """
    target = artifacts_dir / result.versions.preprocessing_version
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, payload in sorted(result.artifacts().items()):
        path = target / name
        path.write_bytes(canonical_bytes(payload))
        written.append(path)

        readable = target / f"readable_{name}"
        readable.write_bytes(
            (json.dumps(json.loads(canonical_text(payload)), ensure_ascii=False, indent=2) + "\n")
            .encode("utf-8")
        )
    return written


def _with_schema(configs: FrozenConfigs, schema: FeatureSchema) -> FrozenConfigs:
    """Тот же комплект конфигов с обновлённой схемой.

    Схема заменяется целиком, а не правится на месте: `resolve_bucket_domains`
    возвращает новый объект, и замороженный артефакт остаётся неизменяемым.
    """
    return FrozenConfigs(
        registry=configs.registry,
        identity=configs.identity,
        timestamps=configs.timestamps,
        dedup=configs.dedup,
        events=configs.events,
        sessionization=configs.sessionization,
        schema=schema,
        missing=configs.missing,
        categories=configs.categories,
        fx=configs.fx,
        bucketization=configs.bucketization,
        time_delta=configs.time_delta,
        profile=configs.profile,
    )


def _golden_vectors_version(dataset: TrainDataset) -> str:
    """§30: версия golden-набора живёт в манифесте датасета, не в конфигах."""
    version = dataset.manifest.get("golden_vectors_version")
    if not version:
        raise BuildPhaseError(
            f"{dataset.identifier}: в манифесте нет golden_vectors_version (§30)"
        )
    return str(version)
