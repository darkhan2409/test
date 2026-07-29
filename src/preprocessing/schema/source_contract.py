"""Source Contract — §4.

Контракт описывает источник целиком, а читает его не только `SourceReader`:
`source_priority` нужен §13, `timezone` и `event_time` — §12, `correction_reversal`
— §9, `late_arriving_policy` — §33.12. Поэтому модель живёт рядом с остальными
декларативными схемами, а не внутри компонента чтения.

Проверки контракта делаются один раз при загрузке, а не на каждой записи:
ошибка в конфиге обязана падать на старте прогона, а не через час обработки
в виде странного `KeyError`.

YAML в хэш состояния не идёт. `state()` отдаёт провалидированную модель в
JSON-совместимом виде — её и сериализует §29.1.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceKind(StrEnum):
    """Что источник поставляет.

    Разделение нужно потому, что §4 требует `event time` у каждого контракта,
    а у справочника курсов события отсутствуют как класс: там есть дата
    действия курса. Одинаковое поле с разным смыслом честнее пометить, чем
    делать вид, что курс — это событие клиента.
    """

    EVENT = "event"
    PROFILE = "profile"
    REFERENCE = "reference"


class ColumnType(StrEnum):
    """Типы колонок — ровно те, что встречаются в источниках.

    Список намеренно короткий: тип, которого нет в данных, проверить нельзя,
    а неиспользуемая ветка валидации создаёт ложное чувство покрытия.
    """

    STRING = "string"
    INTEGER = "integer"
    STRING_LIST = "string_list"


class PiiClass(StrEnum):
    """PII classification (§4). Значение обязательно у каждой колонки:
    умолчание «не PII» — ровно та тихая ошибка, которую §4 и запрещает."""

    NONE = "none"
    PSEUDONYMOUS_ID = "pseudonymous_id"
    QUASI_IDENTIFIER = "quasi_identifier"
    SENSITIVE = "sensitive"
    DIRECT_IDENTIFIER = "direct_identifier"


class TimestampKind(StrEnum):
    """Как источник записывает время (§12 п.1 — «парсится строго по контракту»)."""

    NAIVE_LOCAL_PATTERN = "naive_local_pattern"
    ISO_UTC = "iso_utc"
    EPOCH_MILLIS = "epoch_millis"
    DATE_ISO = "date_iso"


class TimezonePolicy(StrEnum):
    """Откуда берётся IANA-зона (§12 пп. 3, 6, 7)."""

    FROM_FIELD_MAPPING = "from_field_mapping"
    CLIENT_PROFILE_REGION = "client_profile_region"
    SOURCE_DEFAULT = "source_default"
    UTC = "utc"


class UpdateRule(StrEnum):
    """Правила обновления (§4)."""

    APPEND_ONLY = "append_only"
    OVERWRITE_BY_PK = "overwrite_by_pk"
    VERSIONED_BY_FIELD = "versioned_by_field"


class DeleteRule(StrEnum):
    """Правила удаления (§4)."""

    NONE = "none"
    SOFT_DELETE_FLAG = "soft_delete_flag"
    HARD_DELETE = "hard_delete"


class LateArrivingPolicy(StrEnum):
    """Политика опоздавших записей (§4, метрика §33.12)."""

    ACCEPT_AND_MONITOR = "accept_and_monitor"
    QUARANTINE = "quarantine"


class UnknownFieldPolicy(StrEnum):
    """Что делать с полем, которого нет в контракте.

    §4 даёт ровно два выхода: остановить несовместимый pipeline либо
    отправить записи в quarantine. Молча принять — запрещено, поэтому
    третьего значения здесь нет.
    """

    QUARANTINE = "quarantine"
    FAIL = "fail"


class ColumnSpec(BaseModel):
    """Одна колонка источника: тип, гарантия наличия, класс PII."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ColumnType
    required: bool
    """Источник гарантирует наличие ключа. Не путать с `required` в Feature
    Schema (§11): там это «признак нужен модели», и отсутствие даёт MISSING."""

    nullable: bool = False
    """Значение может быть `null`. Это MISSING (§15.2), а не нарушение схемы."""

    pii: PiiClass
    description: str = ""


class TimeFieldSpec(BaseModel):
    """Поле времени и способ его разбора."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    kind: TimestampKind
    pattern: str | None = None

    @model_validator(mode="after")
    def _pattern_matches_kind(self) -> "TimeFieldSpec":
        if self.kind is TimestampKind.NAIVE_LOCAL_PATTERN and not self.pattern:
            raise ValueError(f"{self.field}: для {self.kind} обязателен pattern разбора")
        if self.kind is not TimestampKind.NAIVE_LOCAL_PATTERN and self.pattern:
            raise ValueError(f"{self.field}: pattern имеет смысл только для naive_local_pattern")
        return self


class TimezoneSpec(BaseModel):
    """Правило получения IANA-зоны для источника."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: TimezonePolicy
    field: str | None = None
    mapping: str | None = None
    """Имя справочника «значение поля → IANA-зона». Сам справочник — отдельный
    конфиг (§12), контракт только ссылается на него."""

    default: str | None = None
    """Утверждённый source default (§12 п.7). Его применение — не норма,
    а fallback, и он поднимает `calendar_timezone_fallback_rate` (§33.11)."""

    @model_validator(mode="after")
    def _policy_is_complete(self) -> "TimezoneSpec":
        if self.policy is TimezonePolicy.FROM_FIELD_MAPPING:
            if not self.field or not self.mapping:
                raise ValueError("from_field_mapping требует field и mapping")
        elif self.field or self.mapping:
            raise ValueError(f"{self.policy}: field/mapping имеют смысл только для from_field_mapping")

        needs_default = {TimezonePolicy.CLIENT_PROFILE_REGION, TimezonePolicy.SOURCE_DEFAULT}
        if self.policy in needs_default and not self.default:
            raise ValueError(f"{self.policy} требует утверждённый default (§12 п.7)")
        if self.policy is TimezonePolicy.UTC and self.default:
            raise ValueError("policy utc не нуждается в default")
        return self


class CorrectionReversalSpec(BaseModel):
    """Признак correction/reversal (§4), который читает дедупликатор (§9.3, §9.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supported: bool
    version_field: str | None = None
    reversal_flag_field: str | None = None

    @model_validator(mode="after")
    def _marker_present_when_supported(self) -> "CorrectionReversalSpec":
        markers = (self.version_field, self.reversal_flag_field)
        if self.supported and not any(markers):
            raise ValueError(
                "supported: true без version_field и reversal_flag_field — "
                "§9.3 нечем разрешать конфликтующие дубли"
            )
        if not self.supported and any(markers):
            raise ValueError("маркеры correction/reversal заданы при supported: false")
        return self


class SourceContract(BaseModel):
    """Формальный контракт источника — минимальный состав §4."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""  # подставляется из ключа реестра
    kind: SourceKind
    owner: str = Field(min_length=1)
    format: str
    schema_version: str = Field(min_length=1)
    schema_version_field: str | None = None
    """Колонка, в которой источник сам сообщает версию своей схемы. Если её
    значение разошлось с контрактом — §34 `incompatible_schema_version`."""

    source_priority: int
    """Детерминированный ранг для tie-break (§13). Меньше — раньше в timeline."""

    primary_key: tuple[str, ...] = Field(min_length=1)
    client_field: str | None = None

    event_time: TimeFieldSpec
    processing_time: TimeFieldSpec | None = None
    timezone: TimezoneSpec

    update_rules: UpdateRule
    delete_rules: DeleteRule
    correction_reversal: CorrectionReversalSpec

    max_delay_hours: int = Field(ge=0)
    late_arriving_policy: LateArrivingPolicy
    data_retention_days: int = Field(gt=0)
    unknown_field_policy: UnknownFieldPolicy

    columns: dict[str, ColumnSpec] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _columns_sorted_and_named(cls, value: dict[str, ColumnSpec]) -> dict[str, ColumnSpec]:
        for name in value:
            if not name or name != name.strip():
                raise ValueError(f"имя колонки {name!r} пустое или с пробелами по краям")
        # Порядок ключей в YAML произволен, а в хэш состояния уходит dict —
        # сортируем на входе, чтобы порядок записи в файле ни на что не влиял.
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _references_exist(self) -> "SourceContract":
        known = set(self.columns)

        def require(field_name: str, column: str) -> None:
            if column not in known:
                raise ValueError(f"{field_name}: колонки {column!r} нет в схеме источника")

        for column in self.primary_key:
            require("primary_key", column)
            spec = self.columns[column]
            if not spec.required or spec.nullable:
                raise ValueError(
                    f"primary_key: {column!r} обязан быть required и не nullable — "
                    "иначе у записи не будет lineage"
                )
            if spec.type is not ColumnType.STRING:
                raise ValueError(
                    f"primary_key: {column!r} обязан быть string — "
                    "приведение числа к строке зависит от реализации"
                )

        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError("primary_key содержит повторяющиеся колонки")

        if self.client_field is not None:
            require("client_field", self.client_field)
        elif self.kind is not SourceKind.REFERENCE:
            raise ValueError(f"{self.kind}: client_field обязателен")

        require("event_time", self.event_time.field)
        if self.processing_time is not None:
            require("processing_time", self.processing_time.field)

        if self.timezone.field is not None:
            require("timezone", self.timezone.field)
        if self.schema_version_field is not None:
            require("schema_version_field", self.schema_version_field)
        if self.correction_reversal.version_field is not None:
            require("correction_reversal.version_field", self.correction_reversal.version_field)
        if self.correction_reversal.reversal_flag_field is not None:
            require(
                "correction_reversal.reversal_flag_field",
                self.correction_reversal.reversal_flag_field,
            )

        if (
            self.update_rules is UpdateRule.VERSIONED_BY_FIELD
            and not self.correction_reversal.version_field
        ):
            raise ValueError("versioned_by_field требует correction_reversal.version_field")

        return self


class SourceContractRegistry(BaseModel):
    """Набор контрактов с единой версией (§30 `source_contract_version`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_contract_version: str = Field(min_length=1)
    ignored_paths: tuple[str, ...] = ()
    sources: dict[str, SourceContract] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _inject_names(cls, data: Any) -> Any:
        """Проставить каждому контракту его имя из ключа реестра.

        Дублировать имя внутри контракта — лишний шанс на рассинхрон, поэтому
        в YAML его нет, а модели оно нужно для сообщений об ошибках.
        """
        if not isinstance(data, dict):
            return data
        sources = data.get("sources")
        if not isinstance(sources, dict):
            return data

        named = {}
        for name, contract in sorted(sources.items()):
            if isinstance(contract, dict):
                declared = contract.get("name")
                if declared not in (None, "", name):
                    raise ValueError(f"контракт {name!r} объявляет другое имя: {declared!r}")
                contract = {**contract, "name": name}
            named[name] = contract
        return {**data, "sources": named}

    @model_validator(mode="after")
    def _priorities_unique(self) -> "SourceContractRegistry":
        # Одинаковый ранг у двух источников не ломает детерминизм (третий ключ
        # §13 всё равно решает), но делает порядок случайным по смыслу:
        # он начинает зависеть от формата source_record_id, а не от решения
        # владельца данных. Требуем уникальность — это проектное решение.
        priorities: dict[int, str] = {}
        for name in sorted(self.sources):
            priority = self.sources[name].source_priority
            if priority in priorities:
                raise ValueError(
                    f"source_priority {priority} у двух источников: "
                    f"{priorities[priority]} и {name} (§13)"
                )
            priorities[priority] = name
        return self

    def contract(self, source: str) -> SourceContract:
        try:
            return self.sources[source]
        except KeyError:
            raise KeyError(
                f"у источника {source!r} нет Source Contract (§4); известны: "
                + ", ".join(sorted(self.sources))
            ) from None

    def source_priority(self) -> dict[str, int]:
        """Ранги для tie-break §13."""
        return {name: contract.source_priority for name, contract in self.sources.items()}

    def state(self) -> dict[str, Any]:
        """Контракты в JSON-совместимом виде — вход для §29.1 и §30.

        `mode="json"` здесь обязателен: enum и tuple в канонической
        сериализации запрещены как неявное приведение типов.
        """
        return self.model_dump(mode="json")


def load_source_contracts(path: Path) -> SourceContractRegistry:
    """Загрузить и проверить контракты.

    `safe_load` — не осторожность ради осторожности: полный загрузчик YAML
    умеет конструировать произвольные объекты, а конфиг обязан оставаться
    данными.
    """
    if not path.exists():
        raise FileNotFoundError(f"нет файла Source Contracts: {path}")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: ожидался YAML-объект, получено {type(document).__name__}")

    return SourceContractRegistry.model_validate(document)
