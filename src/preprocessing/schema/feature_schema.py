"""Feature Schema — §11 и §11.2.

Схема — общий контракт препроцессинга и токенайзера, поэтому её инварианты
стоит проверять здесь, а не обнаруживать при сборке словаря:

- `event_type` — атрибут верхнего уровня и в списке `fields` его нет (§11);
- bucket-поле обязано быть `bucket_closed_set`, иначе редкий хвостовой бакет
  схлопнется в `RARE` (§11.1, §14.1 токенайзера);
- у closed-set поля обязан быть полный domain, и в нём обязан быть `MISSING`:
  токенайзер включает все значения domain в словарь независимо от частоты;
- **одноимённое поле в разных типах событий обязано иметь одинаковые тип,
  политику и domain.** Причина не очевидна из §11: токен значения имеет вид
  `value:<field>:<value>` (§14.4.1.1 токенайзера) — тип события в него не
  входит. Два разных domain у одного имени поля дали бы один и тот же токен
  для разных сущностей.

Параметры бакетизации (метод, число бакетов) сюда сознательно не заведены:
их форма определится на шаге 2.13 вместе с `Bucketizer`, и угадывать её
заранее — значит переделывать схему потом.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import MISSING, PROFILE_SECTION, RESERVED_FIELD_NAMES


class VocabularyPolicy(StrEnum):
    """§11: как токенайзер обходится со значениями поля."""

    CLOSED_SET = "closed_set"
    BUCKET_CLOSED_SET = "bucket_closed_set"
    FREQUENCY_PRUNED = "frequency_pruned"
    EXCLUDED = "excluded"


class FieldType(StrEnum):
    """Тип подготовленного значения.

    Текстового типа нет намеренно: §23 запрещает свободный текст в базовой
    версии архитектуры, и запрет выражен отсутствием возможности его объявить.
    """

    CATEGORICAL = "categorical"
    BUCKET = "bucket"


class HighCardinalityPolicy(StrEnum):
    """§22: что делать с полем большой мощности.

    Перечислены только реализованные стратегии. `replace_with_category`,
    `keep_frequent` и `hashing` из §22 сюда не внесены: объявить стратегию,
    которой нет, значит получить пустое место в данных вместо отказа при
    загрузке конфига.
    """

    EXCLUDE = "exclude"
    """Технический ID вообще не доходит до модели (§22 стратегия 1)."""

    PASS_TO_TOKENIZER = "pass_to_tokenizer"
    """Нормализованное значение уходит токенайзеру, решение о `RARE` — за ним
    (§22 стратегия 5). Препроцессинг `RARE` не назначает."""


CLOSED_POLICIES = frozenset({VocabularyPolicy.CLOSED_SET, VocabularyPolicy.BUCKET_CLOSED_SET})


class NumericType(StrEnum):
    """§17: числовой тип поля."""

    INTEGER = "integer"
    DECIMAL = "decimal"


class ParsingLocale(StrEnum):
    """§17: как источник записывает число.

    `RU_KZ` — разделитель тысяч пробелом, дробная часть запятой
    (`"15 000,50"`). `PLAIN` — без разделителей, точка (`"15000.50"`).
    """

    RU_KZ = "ru_KZ"
    PLAIN = "plain"


class NumericSpec(BaseModel):
    """Числовые атрибуты поля — перечень §17.

    Метод бакетизации и clipping policy здесь не заданы: §17 их упоминает,
    но считает их §19, и их форма определится вместе с `Bucketizer` (2.13).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    numeric_type: NumericType
    signed: bool
    """§17.3: допустимо ли отрицательное значение. Для баланса — да,
    для возраста — нет. Правило задаётся полю, а не типу."""

    unit: str = Field(min_length=1)
    """Единица итогового значения: `KZT`, `seconds`, `count`, `years`."""

    scale: Decimal = Decimal(1)
    """Множитель к `unit`: сумма в тиынах приходит со `scale = 0.01`."""

    locale: ParsingLocale = ParsingLocale.PLAIN
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    """Business-valid range (§17.2). Значение вне него — невалидное, то есть
    `MISSING`. Не путать с clipping (§19.4): там значение вне TRAIN-диапазона
    прижимается к крайнему бакету, а здесь оно бессмысленно по существу."""

    @model_validator(mode="after")
    def _range_is_sane(self) -> "NumericSpec":
        if self.scale <= 0:
            raise ValueError("scale обязан быть положительным")
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("min_value больше max_value")
        if not self.signed and self.min_value is not None and self.min_value < 0:
            raise ValueError("unsigned-поле не может иметь отрицательный min_value")
        return self


class FieldSpec(BaseModel):
    """Описание одного поля (§11)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    """Ключ в сырой записи, из которого поле берётся.

    Живёт на уровне типа события, а не имени поля: `amount_base_bucket`
    приходит из `amount` в ядре платежей и из `amount_minor` в карточном
    процессинге. В `shared_signature` не входит по той же причине, что и
    `priority` — источник значения на токен не влияет.
    """

    type: FieldType
    vocabulary_policy: VocabularyPolicy
    required: bool = False
    """Применимо ли поле к каждой записи этого типа события (§15.1).

    `true` — применимо всегда: значения нет → `MISSING`.
    `false` — применимо не к каждой записи: ключа нет → поля нет вовсе.
    Регламент этой разницы не проговаривает; трактовка выведена из §15.1,
    где «неприменимо» и «нет значения» дают разный результат.
    """

    domain: tuple[str, ...] | None = None
    priority: int = Field(ge=1)

    computed: bool = False
    """Значение не читается из сырой записи, а считается компонентом цепочки
    (§24 life-long признаки). Для таких полей `source_field` — имя, под
    которым значение кладут, а не колонка источника."""

    multivalue: bool = False
    order_significant: bool | None = None
    """§21 пп. 1–3: значим ли порядок значений. Обязателен у многозначного
    поля и запрещён у остальных: «не указано» здесь означало бы, что решение
    о сортировке принимает не человек, а порядок в источнике."""

    max_values_per_field: int | None = Field(default=None, gt=0)
    model_input: bool = True
    fx_normalized: bool = False
    """§18.3: проходит ли поле историческую FX-нормализацию.

    От этого зависит имя: FX-нормализованные суммы называются
    `<field>_base_bucket`, прочие числовые поля — `<field>_bucket`.
    """
    currency_field: str | None = None
    """Имя поля схемы, где лежит исходная валюта суммы (§18).

    Обязательно у FX-нормализуемых полей: без него неизвестно, из чего
    пересчитывать, а угадать по имени — верный способ однажды пересчитать
    тенге по курсу доллара.
    """
    numeric: NumericSpec | None = None
    high_cardinality: HighCardinalityPolicy | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "FieldSpec":
        if self.name in RESERVED_FIELD_NAMES:
            raise ValueError(f"поле {self.name!r} зарезервировано и не может быть в Feature Schema")

        if self.type is FieldType.BUCKET and self.vocabulary_policy is not VocabularyPolicy.BUCKET_CLOSED_SET:
            raise ValueError(
                f"{self.name}: bucket-поле обязано иметь vocabulary_policy=bucket_closed_set "
                "(§11.1), иначе редкий бакет схлопнется в RARE"
            )

        if self.vocabulary_policy is VocabularyPolicy.CLOSED_SET and not self.domain:
            raise ValueError(
                f"{self.name}: у closed-set поля обязан быть полный domain — токенайзер "
                "включает его в словарь целиком (§11.1). Это справочник на входе, "
                "а не результат обработки"
            )

        if self.domain is not None:
            if MISSING not in self.domain:
                raise ValueError(
                    f"{self.name}: domain обязан содержать {MISSING} — значение пропуска входит "
                    "в словарь всегда (§14.1 токенайзера)"
                )
            if len(set(self.domain)) != len(self.domain):
                raise ValueError(f"{self.name}: domain содержит повторяющиеся значения")
            if any(not value for value in self.domain):
                raise ValueError(f"{self.name}: domain содержит пустое значение")

        if self.vocabulary_policy is VocabularyPolicy.EXCLUDED and self.model_input:
            raise ValueError(f"{self.name}: excluded-поле не может быть model input")

        if not self.multivalue and self.max_values_per_field is not None:
            raise ValueError(
                f"{self.name}: max_values_per_field имеет смысл только у многозначного поля (§21)"
            )

        if self.multivalue and self.order_significant is None:
            raise ValueError(
                f"{self.name}: у многозначного поля обязан быть объявлен order_significant "
                "(§21 пп. 1–3) — иначе порядок значений определит источник, а не решение"
            )
        if not self.multivalue and self.order_significant is not None:
            raise ValueError(
                f"{self.name}: order_significant имеет смысл только у многозначного поля"
            )

        if self.type is FieldType.BUCKET and self.numeric is None:
            raise ValueError(
                f"{self.name}: у bucket-поля обязан быть блок numeric — §17 требует "
                "объявить тип, знак, диапазон, единицу и locale разбора"
            )
        if self.type is not FieldType.BUCKET and self.numeric is not None:
            raise ValueError(f"{self.name}: numeric имеет смысл только у bucket-поля")

        if self.fx_normalized:
            if self.type is not FieldType.BUCKET:
                raise ValueError(f"{self.name}: FX-нормализация применима только к сумме")
            if not self.currency_field:
                raise ValueError(
                    f"{self.name}: FX-нормализуемому полю нужен currency_field — "
                    "иначе неизвестно, из какой валюты пересчитывать (§18)"
                )
        elif self.currency_field:
            raise ValueError(f"{self.name}: currency_field задан без fx_normalized")

        if self.type is FieldType.BUCKET and self.domain is not None:
            # Domain бакет-поля — результат BUILD, а не конфигурация: §11.1
            # говорит «preprocessing публикует полный domain», а §19 считает
            # его границы по TRAIN. Значение, объявленное руками, разошлось бы
            # с посчитанным при первом же изменении числа бакетов.
            raise ValueError(
                f"{self.name}: domain bucket-поля не объявляется в конфиге — "
                "его публикует BUILD по границам §19"
            )
        return self

    def shared(self) -> "SharedFieldSpec":
        """Часть описания, не зависящая от источника значения.

        Ровно эта часть обязана совпадать у одноимённых полей во всех типах
        событий — и ровно её отдаёт `FeatureSchema.field_specs()`.
        """
        return SharedFieldSpec(
            name=self.name,
            type=self.type,
            vocabulary_policy=self.vocabulary_policy,
            domain=self.domain,
            computed=self.computed,
            multivalue=self.multivalue,
            order_significant=self.order_significant,
            max_values_per_field=self.max_values_per_field,
            model_input=self.model_input,
            fx_normalized=self.fx_normalized,
            currency_field=self.currency_field,
            high_cardinality=self.high_cardinality,
            unit=self.numeric.unit if self.numeric else None,
        )


class SharedFieldSpec(BaseModel):
    """Описание поля без привязки к типу события.

    Существует затем, чтобы неправильное использование стало невозможным,
    а не задокументированным. `source_field`, `numeric`, `priority` и
    `required` описывают, **откуда** берётся значение, и законно различаются
    между типами событий: `amount_base_bucket` приходит строкой в тенге из
    ядра платежей и целым в тиынах из карточного процессинга. Взять их из
    «первого попавшегося» описания — тихо разобрать половину данных неверно;
    так уже случилось однажды.

    Поэтому этих полей здесь просто нет: обращение к ним даёт `AttributeError`
    на месте, а не расхождение в числах через два компонента. Кому нужен
    источник значения — берёт `FeatureSchema.section_specs(...)`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: FieldType
    vocabulary_policy: VocabularyPolicy
    domain: tuple[str, ...] | None
    computed: bool
    multivalue: bool
    order_significant: bool | None
    max_values_per_field: int | None
    model_input: bool
    fx_normalized: bool
    currency_field: str | None
    high_cardinality: HighCardinalityPolicy | None
    unit: str | None
    """Единица итогового значения (§17). После применения `scale` она обязана
    совпадать у одноимённых полей: §19 считает границы бакетов по имени поля,
    и смесь тенге с тиынами дала бы бессмысленный `bucket_17`."""

    def agreement_key(self) -> tuple[object, ...]:
        """Всё, кроме имени, — то, что сверяется между типами событий."""
        return tuple(
            value for key, value in sorted(self.model_dump().items()) if key != "name"
        )


class EventFeatureSchema(BaseModel):
    """Схема полей одного типа события (§11)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(min_length=1)
    fields: tuple[FieldSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> "EventFeatureSchema":
        names = [spec.name for spec in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.event_type}: повторяющиеся имена полей")

        priorities = [spec.priority for spec in self.fields]
        if len(set(priorities)) != len(priorities):
            raise ValueError(
                f"{self.event_type}: повторяющиеся priority — порядок полей стал бы "
                "неоднозначным, а он определяет и порядок токенов, и что обрежется первым (§17)"
            )
        return self

    def ordered_fields(self) -> tuple[FieldSpec, ...]:
        """Поля в порядке Feature Schema — том самом, в котором токенайзер
        эмитит их после `event_type` (§7 токенайзера)."""
        return tuple(sorted(self.fields, key=lambda spec: spec.priority))


class ProfileFeatureSchema(BaseModel):
    """Секция `PROFILE` (§11.2).

    Профильные поля описываются теми же атрибутами, что событийные. Без этого
    правила токенайзерное «`MISSING` для всех полей» и исключение из
    `min_count` к профилю просто неприменимы.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: tuple[FieldSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> "ProfileFeatureSchema":
        names = [spec.name for spec in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("PROFILE: повторяющиеся имена полей")
        priorities = [spec.priority for spec in self.fields]
        if len(set(priorities)) != len(priorities):
            raise ValueError("PROFILE: повторяющиеся priority")
        return self

    def ordered_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(sorted(self.fields, key=lambda spec: spec.priority))


class FeatureSchema(BaseModel):
    """Полная схема: события плюс профиль."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    events: dict[str, EventFeatureSchema] = Field(min_length=1)
    profile: ProfileFeatureSchema

    @model_validator(mode="after")
    def _check_schema(self) -> "FeatureSchema":
        for key, event_schema in self.events.items():
            if key != event_schema.event_type:
                raise ValueError(
                    f"ключ {key!r} не совпадает с event_type {event_schema.event_type!r}"
                )

        shared: dict[str, SharedFieldSpec] = {}
        owners: dict[str, str] = {}
        for owner, spec in self._iter_specs():
            view = spec.shared()
            known = shared.get(spec.name)
            if known is None:
                shared[spec.name] = view
                owners[spec.name] = owner
                continue
            if known.agreement_key() != view.agreement_key():
                differing = sorted(
                    key
                    for key, value in view.model_dump().items()
                    if key != "name" and known.model_dump()[key] != value
                )
                raise ValueError(
                    f"поле {spec.name!r} описано по-разному в {owners[spec.name]} и {owner} "
                    f"({', '.join(differing)}): токен значения имеет вид "
                    "value:<field>:<value> и про тип события не знает, а границы бакетов §19 "
                    "считаются по одному имени поля"
                )
        return self

    def _iter_specs(self) -> Iterator[tuple[str, FieldSpec]]:
        for event_type in sorted(self.events):
            for spec in self.events[event_type].fields:
                yield event_type, spec
        for spec in self.profile.fields:
            yield PROFILE_SECTION, spec

    def field_specs(self) -> dict[str, SharedFieldSpec]:
        """Поля схемы по имени — только то, что одинаково у всех типов событий.

        Возвращает `SharedFieldSpec`, а не `FieldSpec`: у него нет ни
        `source_field`, ни `numeric`, ни `priority`, ни `required`, поэтому
        достать отсюда описание источника значения нельзя даже случайно.
        Для источника есть `section_specs`.
        """
        specs: dict[str, SharedFieldSpec] = {}
        for _, spec in self._iter_specs():
            specs.setdefault(spec.name, spec.shared())
        return dict(sorted(specs.items()))

    def section_specs(self, section: str) -> dict[str, FieldSpec]:
        """Поля одной секции схемы: типа события или `PROFILE`.

        Единственный корректный способ добраться до `numeric` и
        `source_field`: сумма в ядре платежей приходит строкой в тенге, а в
        карточном процессинге — целым в тиынах, и разбирать их одинаково
        нельзя.
        """
        if section == PROFILE_SECTION:
            return {spec.name: spec for spec in self.profile.fields}
        schema = self.events.get(section)
        if schema is None:
            raise KeyError(f"в Feature Schema нет секции {section!r}")
        return {spec.name: spec for spec in schema.fields}

    def closed_set_domains(self) -> dict[str, tuple[str, ...]]:
        """Пункт 4 контракта §2: допустимые значения closed-set полей."""
        return {
            name: spec.domain
            for name, spec in self.field_specs().items()
            if spec.vocabulary_policy is VocabularyPolicy.CLOSED_SET and spec.domain
        }

    def bucket_field_domains(self) -> dict[str, tuple[str, ...]]:
        """Пункт 5 контракта §2: полный configured domain каждого bucket-поля.

        Передаётся токенайзеру как whitelist независимо от TRAIN-частоты —
        именно это спасает хвостовой бакет от `min_count` (§2.2).
        """
        return {
            name: spec.domain
            for name, spec in self.field_specs().items()
            if spec.vocabulary_policy is VocabularyPolicy.BUCKET_CLOSED_SET and spec.domain
        }

    def field_priority(self) -> dict[str, dict[str, int]]:
        """Пункт 12 контракта §2: приоритеты полей по типам событий и профилю."""
        priorities = {
            event_type: {spec.name: spec.priority for spec in schema.ordered_fields()}
            for event_type, schema in sorted(self.events.items())
        }
        priorities[PROFILE_SECTION] = {spec.name: spec.priority for spec in self.profile.ordered_fields()}
        return priorities

    def resolve_bucket_domains(self, domains: dict[str, tuple[str, ...]]) -> "FeatureSchema":
        """Подставить посчитанные на BUILD domain бакет-полей (§11.1, §19).

        Возвращает новую схему: замороженный артефакт не правится на месте.

        Сборка идёт в обход валидаторов сознательно. Правило «domain бакет-поля
        не объявляется» относится к **конфигу**: там его писать руками нельзя.
        После BUILD domain обязан быть заполнен, и та же схема, пропущенная
        через валидацию заново, была бы отвергнута собственным правилом.
        """
        known = {name for name, spec in self.field_specs().items() if spec.type is FieldType.BUCKET}
        missing = sorted(known - set(domains))
        if missing:
            raise ValueError("BUILD не опубликовал domain для полей: " + ", ".join(missing))
        extra = sorted(set(domains) - known)
        if extra:
            raise ValueError("опубликованы domain для не-bucket полей: " + ", ".join(extra))

        def rebuild(spec: FieldSpec) -> FieldSpec:
            if spec.type is not FieldType.BUCKET:
                return spec
            # `model_copy`, а не `model_construct(**model_dump())`:
            # `model_dump` разворачивает вложенный `numeric` в словарь, и
            # у собранного так объекта `spec.numeric.unit` уже не читается.
            return spec.model_copy(update={"domain": tuple(domains[spec.name])})

        return FeatureSchema.model_construct(
            version=self.version,
            events={
                event_type: EventFeatureSchema.model_construct(
                    event_type=event_type,
                    fields=tuple(rebuild(spec) for spec in schema.fields),
                )
                for event_type, schema in self.events.items()
            },
            profile=ProfileFeatureSchema.model_construct(
                fields=tuple(rebuild(spec) for spec in self.profile.fields)
            ),
        )

    def unresolved_bucket_fields(self) -> list[str]:
        """Бакет-поля, у которых domain ещё не опубликован BUILD."""
        return sorted(
            name
            for name, spec in self.field_specs().items()
            if spec.type is FieldType.BUCKET and not spec.domain
        )

    def max_values_per_field(self) -> dict[str, int]:
        """Пункт 13 контракта §2: лимиты многозначных полей, заданные явно."""
        return {
            name: spec.max_values_per_field
            for name, spec in self.field_specs().items()
            if spec.max_values_per_field is not None
        }

    def numeric_rules(self) -> dict[str, dict[str, Any]]:
        """Числовые правила §17 — отдельный пункт перечня §30.

        Разложены по секциям, а не по имени поля: `numeric` в
        `shared_signature` не входит, и одно и то же поле разбирается
        по-разному в разных источниках — сумма в ядре платежей приходит
        строкой в тенге, в карточном процессинге целым в тиынах.
        """
        rules: dict[str, dict[str, Any]] = {}
        for section in (*sorted(self.events), PROFILE_SECTION):
            block = {
                name: spec.numeric.model_dump(mode="json")
                for name, spec in sorted(self.section_specs(section).items())
                if spec.numeric is not None
            }
            if block:
                rules[section] = block
        return rules

    def state(self) -> dict[str, Any]:
        """Состояние для §30. Схема входит целиком: любое её изменение
        меняет состав `fields` у токенайзера."""
        return self.model_dump(mode="json")
