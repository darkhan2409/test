"""ArtifactHasher — §29.1, §30.

Один вопрос, на который отвечает этот модуль: **тем ли состоянием посчитаны
данные, что лежит рядом с ними**. Ответ — `preprocessing_state_sha256`: SHA-256
по канонически сериализованному состоянию препроцессинга. Совпал — артефакты
и данные из одного BUILD; не совпал — обработка останавливается (§30).

Три решения определяют, что этот хэш вообще что-то значит.

**Хэшируется разобранное состояние, а не текст конфигов.** Переставленные
строки YAML, изменённый комментарий и другой отступ не меняют ни одного
решения пайплайна, а хэш по тексту менялся бы от каждого из них — и очень
быстро перестал бы значить «состояние другое». Поэтому в пре-образ идёт то,
что получилось после разбора и валидации.

**Полнота выражена сигнатурой, а не проверкой.** `PreprocessingState` —
frozen dataclass без единого значения по умолчанию, и все разделы §30 в нём
обязательные. Забыть раздел нельзя: это `TypeError` при сборке, а не тихо
неполный хэш. Умолчание, добавленное позже, вернуло бы возможность забыть,
поэтому их отсутствие проверяется при импорте модуля.

**Разделы названы по §30 дословно.** Часть из них — проекции соседей
(`closed_set_domains` и `numeric_rules` выводятся из Feature Schema,
`bucket_field_domains` — из `bucket_edges`). Дублирование внутри пре-образа
безвредно: оба вида берутся из одного объекта в один момент и разойтись не
могут. Зато перечень §30 сверяется с кодом глазами за минуту, а не выводится
из чтения трёх модулей.

Несовпадение хэша — блокирующая ошибка, и вариант «продолжить с
предупреждением» здесь не предусмотрен: `verify_state_hash` ничего не
возвращает, проверять её результат нечем, а значит и проигнорировать нечего.
"""

from __future__ import annotations

from dataclasses import MISSING as NO_DEFAULT
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Mapping, Protocol, runtime_checkable

from .bucketizer import BucketEdges, BucketizationConfig
from .core.canonical import SERIALIZATION_CONFIG, canonical_bytes
from .core.hashing import HASH_POLICY, content_hex
from .core.settings import PreprocessingSettings
from .core.versions import PreprocessingVersions
from .cutoff import cutoff_policy_state
from .field_policies import TEXT_POLICY
from .schema.feature_schema import FeatureSchema
from .timestamp_normalizer import TimestampPolicy

# Перечень содержимого `preprocessing_state_sha256` из §30, дословно и в
# порядке регламента. Сверяется с полями `PreprocessingState` при импорте:
# пункт, выпавший из состояния, не должен уезжать в релиз незамеченным.
SPEC_30_SECTIONS: tuple[str, ...] = (
    "source_contracts",
    "identity_mapping",
    "event_mapping",
    "category_mapping",
    "feature_schema",
    "closed_set_domains",
    "bucket_field_domains",
    "timestamp_policy",
    "calendar_timezone_policy",
    "dedup_config",
    "sessionization_config",
    "fx_config",
    "bucket_edges",
    "time_delta_edges",
    "numeric_rules",
    "text_policy",
    "cutoff_policy",
    "serialization_config",
)


class StateHashMismatchError(RuntimeError):
    """Состояние не то, которым посчитаны артефакты (§30).

    Всегда блокирующая: продолжать обработку значит смешивать данные из двух
    разных BUILD, а расхождение вскроется где-нибудь на границах бакетов.
    """


@runtime_checkable
class StateArtifact(Protocol):
    """Артефакт, умеющий отдать своё состояние для §30.

    Вид состояния определяет владелец артефакта, а не хэшер: границы бакетов
    хранятся строками десятичных дробей, и знать об этом должен §19, а не §30.
    """

    def state(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PreprocessingState:
    """Хэшируемое состояние препроцессинга — перечень §30 плюс обоснованные
    дополнения.

    Значений по умолчанию здесь нет ни у одного поля. Это и есть гарантия
    полноты: раздел нельзя не передать.
    """

    # --- перечень §30 ---
    source_contracts: Mapping[str, Any]
    identity_mapping: Mapping[str, Any]
    event_mapping: Mapping[str, Any]
    category_mapping: Mapping[str, Any]
    feature_schema: Mapping[str, Any]
    closed_set_domains: Mapping[str, Any]
    bucket_field_domains: Mapping[str, Any]
    timestamp_policy: Mapping[str, Any]
    calendar_timezone_policy: Mapping[str, Any]
    dedup_config: Mapping[str, Any]
    sessionization_config: Mapping[str, Any]
    fx_config: Mapping[str, Any]
    bucket_edges: Mapping[str, Any]
    time_delta_edges: Mapping[str, Any]
    numeric_rules: Mapping[str, Any]
    text_policy: Mapping[str, Any]
    cutoff_policy: Mapping[str, Any]
    serialization_config: Mapping[str, Any]

    # --- сверх §30, каждое со своей причиной ---

    hash_policy: Mapping[str, Any]
    """§29.1 п.13 требует этого прямо: формат пре-образа и формула seed входят
    в состояние, иначе их изменение прошло бы незамеченным."""

    bucketization_config: Mapping[str, Any]
    """Метод и запрошенное число бакетов. Сами границы уже в состоянии, но по
    ним не восстановить, чем они были посчитаны: §19.3 удаляет совпавшие
    границы, и 32 запрошенных бакета неотличимы от 8 по результату."""

    profile_policy: Mapping[str, Any]
    """§6 и §24: выбор снимка и набор life-long признаков определяют
    `prepared_profile`, а в перечне §30 профиль не назван вовсе."""

    run_policy: Mapping[str, Any]
    """Доменная политика прогона: `global_seed`, `T`, `fx_max_staleness`,
    разрыв сессии, лимит многозначных полей, размер выборки. §31 требует
    хранить `global_seed` и `fx_max_staleness`, а меняют они выход целиком."""

    def document(self) -> dict[str, Any]:
        """Состояние как единый документ. Порядок ключей не важен —
        каноническая сериализация всё равно отсортирует их по байтам UTF-8."""
        return {item.name: getattr(self, item.name) for item in dataclass_fields(self)}

    def canonical_bytes(self) -> bytes:
        """Ровно те байты, по которым считается хэш и которые уходят в
        артефакт (§29.1 пп. 1–8)."""
        return canonical_bytes(self.document())

    def sha256(self) -> str:
        """`preprocessing_state_sha256` — §30."""
        return content_hex(self.canonical_bytes())


def build_state(
    *,
    source_contracts: StateArtifact,
    identity_mapping: StateArtifact,
    event_mapping: StateArtifact,
    category_mapping: StateArtifact,
    feature_schema: FeatureSchema,
    timestamp_policy: TimestampPolicy,
    dedup_policy: StateArtifact,
    sessionization: StateArtifact,
    fx_config: StateArtifact,
    profile_policy: StateArtifact,
    bucketization: BucketizationConfig,
    bucket_edges: BucketEdges,
    time_delta_edges: StateArtifact,
    settings: PreprocessingSettings,
) -> PreprocessingState:
    """Собрать состояние §30 из загруженных конфигов и артефактов BUILD.

    Артефакты BUILD (`bucket_edges`, `time_delta_edges`) — обязательные
    параметры, а не «если уже посчитаны». Состояние без них описывало бы не
    тот пайплайн, которым обработаны данные, и хэш обещал бы больше, чем
    проверяет.
    """
    return PreprocessingState(
        source_contracts=source_contracts.state(),
        identity_mapping=identity_mapping.state(),
        event_mapping=event_mapping.state(),
        category_mapping=category_mapping.state(),
        feature_schema=feature_schema.state(),
        closed_set_domains={
            name: list(values) for name, values in feature_schema.closed_set_domains().items()
        },
        # Domain бакет-полей берётся у BUILD, а не у схемы: §11.1 отдаёт
        # публикацию границам §19, и в конфиге его объявить нельзя.
        bucket_field_domains={
            name: list(values) for name, values in bucket_edges.bucket_field_domains().items()
        },
        timestamp_policy=timestamp_policy.state(),
        calendar_timezone_policy=_calendar_timezone_policy(timestamp_policy),
        dedup_config=dedup_policy.state(),
        sessionization_config=sessionization.state(),
        fx_config=fx_config.state(),
        bucket_edges=bucket_edges.state(),
        time_delta_edges=time_delta_edges.state(),
        numeric_rules=feature_schema.numeric_rules(),
        text_policy=dict(TEXT_POLICY),
        cutoff_policy=cutoff_policy_state(settings.cutoff_time),
        serialization_config=dict(SERIALIZATION_CONFIG),
        hash_policy=dict(HASH_POLICY),
        bucketization_config=bucketization.state(),
        profile_policy=profile_policy.state(),
        run_policy=settings.policy_state(),
    )


def _calendar_timezone_policy(policy: TimestampPolicy) -> dict[str, Any]:
    """§30 называет календарную политику отдельным пунктом, а §12 описывает её
    вместе с разбором времени — здесь она выделена проекцией.

    Разделять сам конфиг незачем: обе половины версионируются раздельно
    (`timestamp_policy_version` и `calendar_timezone_policy_version`), а живут
    в одном файле, потому что справочник зон нужен обеим.
    """
    return {
        "calendar_timezone_policy_version": policy.calendar_timezone_policy_version,
        "timezone_mappings": {
            name: dict(sorted(values.items()))
            for name, values in sorted(policy.timezone_mappings.items())
        },
        "client_profile_region": policy.client_profile_region.model_dump(mode="json"),
    }


def verify_state_hash(state: PreprocessingState, expected: str) -> None:
    """Сверить состояние с ожидаемым хэшем (§30, «hash проверяется при загрузке»).

    Ничего не возвращает и ничего не логирует: единственный исход, кроме
    молчаливого успеха, — исключение. «Продолжить с предупреждением» здесь
    не выражается.
    """
    actual = state.sha256()
    if actual != expected:
        raise StateHashMismatchError(
            f"preprocessing_state_sha256 не совпал: ожидался {expected}, посчитан {actual} — "
            "артефакты и данные из разных BUILD, обработка остановлена (§30)"
        )


def with_state_hash(
    versions: PreprocessingVersions, state: PreprocessingState
) -> PreprocessingVersions:
    """Дописать `preprocessing_state_sha256` в реестр версий (§30).

    Возвращает новый реестр: замороженный комплект версий не правится на
    месте, как и любой другой артефакт BUILD.
    """
    return versions.model_copy(update={"preprocessing_state_sha256": state.sha256()})


def _check_state_is_complete() -> None:
    """Проверки, которые дешевле сделать при импорте, чем искать потом.

    Первая: каждый пункт §30 присутствует полем. Вторая: ни у одного поля нет
    значения по умолчанию — иначе раздел можно было бы не передать, и хэш
    посчитался бы по неполному состоянию, ничем себя не выдав.
    """
    declared = {item.name for item in dataclass_fields(PreprocessingState)}

    missing = sorted(set(SPEC_30_SECTIONS) - declared)
    if missing:
        raise RuntimeError(
            "в состоянии нет разделов, которые перечисляет §30: " + ", ".join(missing)
        )

    optional = sorted(
        item.name
        for item in dataclass_fields(PreprocessingState)
        if item.default is not NO_DEFAULT or item.default_factory is not NO_DEFAULT
    )
    if optional:
        raise RuntimeError(
            "разделы состояния со значением по умолчанию: "
            + ", ".join(optional)
            + " — такой раздел можно не передать, и хэш посчитается по неполному состоянию"
        )


_check_state_is_complete()
