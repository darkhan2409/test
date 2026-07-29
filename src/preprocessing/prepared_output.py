"""Output dataset — §32 и контракт §2.

Здесь препроцессинг заканчивается и начинается чужая ответственность. Всё,
что токенайзер получит, он получит отсюда, и §2 перечисляет это девятнадцатью
пунктами. Пункт, который забыли, обнаружится не здесь, а на сборке словаря —
и будет выглядеть как «странно ведёт себя модель».

**Полнота выражена сигнатурой, как и в §30.** `TokenizerContract` — frozen
dataclass с девятнадцатью обязательными полями, по одному на пункт §2.
Забыть пункт нельзя: это `TypeError` при сборке, а не тихо неполный выход.
Перечень §2 продублирован константой и сверяется с полями при импорте.

**Пункт может не быть отдельным файлом.** `calendar_time_features` и
`lifetime_first` живут внутри записей событий, `fx_max_staleness` и версии —
внутри метаданных. Поэтому пункт описывается парой «файл + поле», а не
именем файла: контракт обязан отвечать на вопрос «где это», а не «есть ли
такой файл».

**Справочные записи в выход не идут.** Курсы валют — материал §18, они не
события и не профиль. До 3.3 они доезжали до конца цепочки и в отчёте
контракта считались профилями; здесь они отсекаются по существу — выход
состоит только из того, что перечисляет §32.

**Lineage сохраняется через `event_id`, а не через `source_meta`.** §32.2
задаёт форму записи события, и сырого `source_record_id` в ней нет — §2.2
прямо не хочет видеть технические ID на входе токенайзера. Связь с сырой
записью не теряется: `event_id` по §8 считается из `source_system`,
`source_record_id`, `event_type` и времени, то есть восстанавливается
пересчётом, а прогонный lineage лежит рядом отдельным файлом.
"""

from __future__ import annotations

from dataclasses import MISSING as NO_DEFAULT
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .core.canonical import canonical_bytes, canonical_text
from .cutoff import cutoff_policy_state
from .feature_projection import ProjectedRecord
from .field_policies import TEXT_POLICY

COMPONENT = "prepared_output"

PROFILE_FILE = "prepared_profile.jsonl"
EVENTS_FILE = "prepared_events.jsonl"
METADATA_FILE = "metadata.json"
CONTRACT_FILE = "contract.json"
LINEAGE_FILE = "lineage.json"

# Перечень §2, дословно и в порядке регламента. Сверяется с полями
# `TokenizerContract` при импорте: пункт, выпавший из выхода, не должен
# уезжать в токенайзер незамеченным.
CONTRACT_ITEMS: tuple[str, ...] = (
    "prepared_profile",
    "prepared_events",
    "feature_schema",
    "closed_set_domains",
    "bucket_field_domains",
    "bucket_metadata",
    "time_delta_edges",
    "calendar_time_features",
    "currency_normalization_config",
    "fx_max_staleness",
    "sessionization_config",
    "field_priority",
    "max_values_per_field",
    "text_policy",
    "cutoff_policy",
    "preprocessing_version",
    "preprocessing_state_sha256",
    "data_quality_statistics",
    "lifetime_first",
)


class PreparedOutputError(RuntimeError):
    """Выход собрать нельзя — блокирующая ошибка."""


class EncodedRun(Protocol):
    """Что этому модулю нужно от результата ENCODE.

    Протокол, а не импорт `EncodeResult`: выход §32 знает, из чего он
    состоит, но не должен знать, как устроена фаза, которая его посчитала.
    Заодно это снимает круговой импорт — `pipeline` вызывает запись выхода,
    а не наоборот.
    """

    artifacts: Any
    configs: Any
    settings: Any
    contract: Any
    metrics: Mapping[str, Any]
    quarantine: Mapping[str, Any]
    metadata: Mapping[str, Any]
    lineage: Mapping[str, Any]

    def events(self) -> list[ProjectedRecord]: ...

    def profiles(self) -> list[ProjectedRecord]: ...


@dataclass(frozen=True)
class ContractItem:
    """Где лежит пункт контракта.

    `field` заполняется, когда пункт — не файл целиком, а часть записей:
    `lifetime_first` это поле каждого события, а не отдельный артефакт.
    """

    file: str
    field: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"file": self.file, "field": self.field}


@dataclass(frozen=True)
class TokenizerContract:
    """Девятнадцать пунктов §2 — по одному полю на пункт, без умолчаний."""

    prepared_profile: ContractItem
    prepared_events: ContractItem
    feature_schema: ContractItem
    closed_set_domains: ContractItem
    bucket_field_domains: ContractItem
    bucket_metadata: ContractItem
    time_delta_edges: ContractItem
    calendar_time_features: ContractItem
    currency_normalization_config: ContractItem
    fx_max_staleness: ContractItem
    sessionization_config: ContractItem
    field_priority: ContractItem
    max_values_per_field: ContractItem
    text_policy: ContractItem
    cutoff_policy: ContractItem
    preprocessing_version: ContractItem
    preprocessing_state_sha256: ContractItem
    data_quality_statistics: ContractItem
    lifetime_first: ContractItem

    def manifest(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name).describe()
            for item in dataclass_fields(self)
        }


def profile_row(record: ProjectedRecord) -> dict[str, Any]:
    """Запись `prepared_profile` в форме §32.1."""
    if record.timestamp_utc is None:
        raise PreparedOutputError(
            f"{record.raw_reference}: профиль без profile_time_utc — §6 обязан был "
            "поставить хотя бы T"
        )
    return {
        "client_id": record.client_id,
        "profile_time_utc": _moment(record.timestamp_utc),
        "fields": dict(sorted(record.fields.items())),
    }


def event_row(record: ProjectedRecord) -> dict[str, Any]:
    """Запись `prepared_events` в форме §32.2 плюс `lifetime_first` (§2 п.19).

    `event_type` только top-level — это §32.2 говорит прямо, а §11 не даёт
    объявить одноимённое поле в схеме, так что дублировать его нечем.
    `delta_from_previous_event` не пишется вовсе: §32.2 запрещает отдавать
    финальную дельту до выбора окна, а поля с таким именем в записи нет.
    """
    features = getattr(record, "calendar_time_features", None)
    if features is None:
        raise PreparedOutputError(
            f"{record.raw_reference}: событие без календарных признаков — §25 не отработал"
        )
    return {
        "client_id": record.client_id,
        "event_id": record.event_id,
        "event_type": record.event_type,
        "timestamp_utc": _moment(record.timestamp_utc),
        "calendar_timezone": record.calendar_timezone,
        "ordering_key": record.ordering_key,
        "fields": dict(sorted(record.fields.items())),
        "calendar_time_features": {
            "hour_of_day_local": features.hour_of_day_local,
            "day_of_week_local": features.day_of_week_local,
        },
        "lifetime_first": bool(getattr(record, "lifetime_first", False)),
    }


def write_prepared(result: EncodedRun, out_dir: Path) -> TokenizerContract:
    """Записать выход §32 и вернуть карту контракта §2.

    Порядок записей задаётся явно, а не наследуется от цепочки: файл читают
    построчно, и порядок в нём — часть байт, которые сравнивают golden-векторы
    (§29.2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    events = sorted(result.events(), key=lambda item: (item.client_id or "", item.position))
    profiles = sorted(result.profiles(), key=lambda item: item.client_id or "")

    _write_lines(out_dir / PROFILE_FILE, [profile_row(item) for item in profiles])
    _write_lines(out_dir / EVENTS_FILE, [event_row(item) for item in events])

    schema = result.configs.schema
    edges = result.artifacts.bucket_edges
    settings = result.settings

    files: dict[str, Any] = {
        "feature_schema.json": schema.state(),
        "closed_set_domains.json": {
            name: list(values) for name, values in schema.closed_set_domains().items()
        },
        "bucket_field_domains.json": {
            name: list(values) for name, values in edges.bucket_field_domains().items()
        },
        "bucket_metadata.json": edges.bucket_metadata(),
        "time_delta_edges.json": result.artifacts.time_delta_edges.state(),
        "currency_normalization_config.json": result.configs.fx.state(),
        "sessionization_config.json": result.configs.sessionization.state(),
        "field_priority.json": schema.field_priority(),
        "max_values_per_field.json": schema.max_values_per_field(),
        "text_policy.json": dict(TEXT_POLICY),
        "cutoff_policy.json": cutoff_policy_state(settings.cutoff_time),
        "data_quality.json": {
            "metrics": result.metrics,
            "quarantine": result.quarantine,
            "output_contract": result.contract.summary(),
        },
        METADATA_FILE: {
            **result.metadata,
            # §2 п.10 — отдельный пункт контракта, но не отдельный файл:
            # это одно число, и жить ему рядом с версиями (§32.3).
            "fx_max_staleness_days": settings.fx_max_staleness_days,
        },
        LINEAGE_FILE: dict(result.lineage),
    }
    for name, payload in sorted(files.items()):
        (out_dir / name).write_bytes(canonical_bytes(payload))

    contract = TokenizerContract(
        prepared_profile=ContractItem(PROFILE_FILE),
        prepared_events=ContractItem(EVENTS_FILE),
        feature_schema=ContractItem("feature_schema.json"),
        closed_set_domains=ContractItem("closed_set_domains.json"),
        bucket_field_domains=ContractItem("bucket_field_domains.json"),
        bucket_metadata=ContractItem("bucket_metadata.json"),
        time_delta_edges=ContractItem("time_delta_edges.json"),
        calendar_time_features=ContractItem(EVENTS_FILE, "calendar_time_features"),
        currency_normalization_config=ContractItem("currency_normalization_config.json"),
        fx_max_staleness=ContractItem(METADATA_FILE, "fx_max_staleness_days"),
        sessionization_config=ContractItem("sessionization_config.json"),
        field_priority=ContractItem("field_priority.json"),
        max_values_per_field=ContractItem("max_values_per_field.json"),
        text_policy=ContractItem("text_policy.json"),
        cutoff_policy=ContractItem("cutoff_policy.json"),
        preprocessing_version=ContractItem(METADATA_FILE, "preprocessing_version"),
        preprocessing_state_sha256=ContractItem(METADATA_FILE, "preprocessing_state_sha256"),
        data_quality_statistics=ContractItem("data_quality.json"),
        lifetime_first=ContractItem(EVENTS_FILE, "lifetime_first"),
    )
    (out_dir / CONTRACT_FILE).write_bytes(canonical_bytes(contract.manifest()))

    _check_contract_is_readable(contract, out_dir)
    return contract


def _write_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """JSONL, где каждая строка канонична (§29.1).

    Файл целиком — не JSON-документ, поэтому канонизируется построчно: так
    его можно читать потоком, не удерживая в памяти, и при этом сравнивать
    побайтно.
    """
    payload = "".join(canonical_text(row) + "\n" for row in rows)
    path.write_bytes(payload.encode("utf-8"))


def _moment(value: Any) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _check_contract_is_readable(contract: TokenizerContract, out_dir: Path) -> None:
    """Каждый пункт контракта указывает на существующий файл.

    Проверка дешёвая и ловит ровно то, чего не ловит сигнатура: поле
    заполнено, но указывает в пустоту — например, файл переименовали, а карту
    не поправили.
    """
    for item in dataclass_fields(contract):
        target = out_dir / getattr(contract, item.name).file
        if not target.exists():
            raise PreparedOutputError(
                f"пункт контракта {item.name} указывает на {target.name}, которого нет"
            )


def _check_contract_is_complete() -> None:
    """Перечень §2 совпадает с полями, и ни у одного нет умолчания.

    Та же пара проверок, что у состояния §30, и по той же причине: умолчание
    вернуло бы возможность не заполнить пункт, а расхождение с перечнем
    означало бы, что контракт с токенайзером живёт в двух версиях сразу.
    """
    declared = {item.name for item in dataclass_fields(TokenizerContract)}

    missing = sorted(set(CONTRACT_ITEMS) - declared)
    if missing:
        raise RuntimeError("в выходе нет пунктов контракта §2: " + ", ".join(missing))

    extra = sorted(declared - set(CONTRACT_ITEMS))
    if extra:
        raise RuntimeError("в контракте есть пункты сверх §2: " + ", ".join(extra))

    optional = sorted(
        item.name
        for item in dataclass_fields(TokenizerContract)
        if item.default is not NO_DEFAULT or item.default_factory is not NO_DEFAULT
    )
    if optional:
        raise RuntimeError(
            "пункты контракта со значением по умолчанию: "
            + ", ".join(optional)
            + " — такой пункт можно не заполнить, и выход уедет неполным"
        )


_check_contract_is_complete()
