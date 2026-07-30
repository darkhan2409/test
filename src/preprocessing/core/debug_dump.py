"""Debug-режим: трассировка вход/выход компонентов.

Цепочка §37.2 длинная — семнадцать компонентов. По конечному
`prepared_events` не видно, на каком шаге значение испортилось: `MISSING` в
`currency` мог поставить и CategoryNormalizer, и FeatureSchemaRegistry, и
NumericValidator. Дамп «вход → выход» каждого шага даёт покомпонентный разрез
одного клиента.

Три правила, которые делают режим безопасным:

1. **Не входит в состояние.** Дампы не участвуют в `preprocessing_state_sha256`
   (§30) и не входят в обязательные артефакты (§31). Это диагностика.
2. **Не меняет поведение.** При `debug=false` вызовы — no-op, и результат
   обязан быть байт-в-байт тем же, что с включённым режимом. Поэтому `record`
   принимает ленивый итератор: выключенный дамп его не потребляет, и стоимость
   формирования строк не платится.
3. **Не растёт бесконтрольно.** 17 компонентов × 2 стадии × 76 тысяч записей —
   сотни мегабайт. Поэтому дамп пишется только для отобранных `client_id`.

Раздел регламента к строке дописывается автоматически из реестра компонентов,
а не передаётся вызывающим кодом: так пометка не разъедется с планом.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

EDGE_CASE_MANIFEST = "edge_case_manifest.json"
IDENTITY_MAPPING = "identity_mapping.json"
DATASET_MANIFEST = "manifest.json"
META_DIR = "_meta"

CLEAN_CLIENTS_KEY = "debug_clean_clients"
"""Поле манифеста: сколько клиентов набор обещает без краевых случаев.

Число приходит с данными, а не константой в этом модуле. На демо главный экран
показывает путь нормальной транзакции, и дамп из одного брака показывать
нечего — но сколько чистых клиентов в наборе есть, знает только генератор, и у
main и golden это может отличаться.
"""

# Ключи привязки строки к записи. Всё остальное из переданного словаря
# уходит в payload.
_IDENTITY_KEYS = frozenset({"client_id", "client_ref", "event_id", "source_record_id"})


class Stage(StrEnum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class Component:
    """Компонент цепочки §37.2 с его местом в порядке обработки."""

    order: int
    name: str
    spec_section: str
    plan_item: str

    @property
    def slug(self) -> str:
        """Имя каталога: номер спереди, чтобы каталоги сортировались
        в порядке обработки, а не по алфавиту."""
        return f"{self.order:02d}_{self.name}"


# Порядок и состав — цепочка §37.2, пункты плана 2.1–2.17.
COMPONENTS: tuple[Component, ...] = (
    Component(1, "source_reader", "§4", "2.1"),
    Component(2, "identity_resolver", "§7", "2.2"),
    Component(3, "timestamp_normalizer", "§12", "2.3"),
    Component(4, "cutoff_filter", "§14", "2.4"),
    Component(5, "deduplicator", "§9", "2.5"),
    Component(6, "event_mapper", "§8, §10", "2.6"),
    Component(7, "sessionizer", "§20", "2.7"),
    Component(8, "feature_schema", "§11, §15", "2.8"),
    Component(9, "category_normalizer", "§16", "2.9"),
    Component(10, "numeric_validator", "§17", "2.10"),
    Component(11, "fx_normalizer", "§18", "2.11"),
    Component(12, "deterministic_sampler", "§19.6", "2.12"),
    Component(13, "bucketizer", "§19", "2.13"),
    Component(14, "field_policies", "§21, §22, §23", "2.14"),
    Component(15, "profile_builder", "§6, §24", "2.15"),
    Component(16, "timeline_builder", "§13, §26", "2.16"),
    Component(17, "time_feature_builder", "§25", "2.17"),
)

COMPONENTS_BY_NAME: dict[str, Component] = {item.name: item for item in COMPONENTS}

# До EventMapper (компонент 6) event_id ещё не существует: его считает именно
# он (§8). Более ранние компоненты привязываются к source_record_id.
EVENT_ID_AVAILABLE_FROM = COMPONENTS_BY_NAME["event_mapper"].order


def _jsonable(value: Any) -> str:
    """Единственное преобразование, которое дампу разрешено делать самому.

    Между §17 и §19 значение поля живёт как `Decimal`, и обычный `json.dumps`
    на нём падает. Печатается оно строкой, а не через `float`: `float` от
    `Decimal("25.99")` даёт `25.989999999999998`, а дамп читают глазами и
    сверяют с выходом.

    Всё остальное — ошибка, а не повод молча позвать `str()`: неожиданный тип
    в дампе означает, что компонент отдал в трассировку не то, что обещал.
    """
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(
        f"{type(value).__name__} не сериализуется в дамп трассировки; "
        "компонент обязан отдавать в debug_row готовые к печати значения"
    )


class DebugDumpError(ValueError):
    """Ошибка вызова трассировки — всегда ошибка в коде компонента."""


class DebugDump:
    """Накопитель дампов. Выключенный экземпляр не делает ничего."""

    def __init__(
        self,
        *,
        enabled: bool,
        debug_dir: Path,
        client_ids: frozenset[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.debug_dir = debug_dir
        self.client_ids = client_ids
        self._rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    @classmethod
    def from_settings(cls, settings: Any, client_ids: frozenset[str] | None = None) -> "DebugDump":
        return cls(enabled=settings.debug, debug_dir=settings.debug_dir, client_ids=client_ids)

    def record(
        self,
        component: str,
        stage: Stage,
        entries: Iterable[Mapping[str, Any]],
    ) -> None:
        """Записать вход или выход компонента.

        `entries` — словари с `client_id` и хотя бы одним из
        `event_id` / `source_record_id`; остальное уходит в `payload`.
        Передавайте генератор: при выключенном режиме он не будет потреблён.
        """
        if not self.enabled:
            return

        slot = COMPONENTS_BY_NAME.get(component)
        if slot is None:
            raise DebugDumpError(
                f"неизвестный компонент {component!r}; допустимые: "
                + ", ".join(sorted(COMPONENTS_BY_NAME))
            )

        bucket = self._rows[(slot.slug, str(stage))]
        for entry in entries:
            row = self._build_row(slot, stage, entry, seq=len(bucket))
            if row is not None:
                bucket.append(row)

    def _build_row(
        self,
        slot: Component,
        stage: Stage,
        entry: Mapping[str, Any],
        *,
        seq: int,
    ) -> dict[str, Any] | None:
        client_id = entry.get("client_id")
        client_ref = entry.get("client_ref")
        event_id = entry.get("event_id")
        source_record_id = entry.get("source_record_id")

        if event_id is None and source_record_id is None:
            raise DebugDumpError(
                f"{slot.slug}/{stage}: строке нужен event_id или source_record_id, "
                "иначе её нельзя связать с сырой записью"
            )
        if slot.order < EVENT_ID_AVAILABLE_FROM and event_id is not None:
            raise DebugDumpError(
                f"{slot.slug}: event_id появляется только на шаге event_mapper (§8), "
                "до него привязка идёт по source_record_id"
            )

        if not self._keep(client_id, client_ref):
            return None

        payload = {
            key: value
            for key, value in entry.items()
            if key not in _IDENTITY_KEYS
        }
        return {
            "seq": seq,
            "component": slot.slug,
            "plan_item": slot.plan_item,
            "spec_section": slot.spec_section,
            "stage": str(stage),
            "client_id": client_id,
            "client_ref": client_ref,
            "event_id": event_id,
            "source_record_id": source_record_id,
            "payload": payload,
        }

    def _keep(self, client_id: str | None, client_ref: str | None) -> bool:
        """Проходит ли строка фильтр по клиентам.

        До IdentityResolver (§7) канонического `client_id` ещё нет — есть
        только ссылка источника (`client_ref`), поэтому фильтр смотрит на обе.
        Строка, у которой нет ни того, ни другого, остаётся: фильтр по клиентам
        не может судить о записи без клиента, а молча выкинуть весь справочник
        курсов — не то поведение, которого ждут от фильтра.
        """
        if self.client_ids is None:
            return True
        known = {value for value in (client_id, client_ref) if value is not None}
        if not known:
            return True
        return bool(known & self.client_ids)

    def merge(self, other: "DebugDump") -> None:
        """Присоединить дамп воркера.

        Порядок строк сохраняется — он и есть порядок обработки, ради которого
        дамп читают. Поэтому вызывающий код обязан сливать воркеров в строгом
        порядке партиций (§29 п.3), как и остальные накопители.
        """
        if not self.enabled:
            return
        for key, rows in other._rows.items():
            self._rows[key].extend(rows)

    def write(self) -> list[Path]:
        """Разложить дампы по каталогам компонентов и вернуть записанные файлы."""
        if not self.enabled:
            return []

        self._clear_previous_run()
        written: list[Path] = []
        for (slug, stage), rows in sorted(self._rows.items()):
            component_dir = self.debug_dir / slug
            component_dir.mkdir(parents=True, exist_ok=True)
            path = component_dir / f"{stage}.jsonl"
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=_jsonable) + "\n"
                for row in rows
            )
            path.write_bytes(payload.encode("utf-8"))
            written.append(path)
        return written

    def _clear_previous_run(self) -> None:
        """Убрать дампы прошлого прогона: иначе рядом с новым разрезом
        останется старый, и разбор поедет."""
        if not self.debug_dir.exists():
            return
        for slot in COMPONENTS:
            component_dir = self.debug_dir / slot.slug
            if not component_dir.exists():
                continue
            for stale in sorted(component_dir.glob("*.jsonl")):
                stale.unlink()


def default_client_filter(
    raw_meta_dir: Path,
    *,
    include_source_refs: bool = True,
) -> frozenset[str]:
    """Клиенты по умолчанию: все с краевыми случаями плюс обещанные «чистые».

    Клиенты с краевыми случаями берутся из `edge_case_manifest.json`, чистые —
    первые по порядку из `identity_mapping.json`, которых там нет. Сколько
    брать чистых, объявляет сам набор полем `debug_clean_clients` в
    `manifest.json`: состав клиентов знает генератор, и у разных наборов число
    может отличаться.

    `include_source_refs` добавляет к набору источниковые ссылки тех же
    клиентов (`client_ref`, `cardholder_id`, `cif`, `login_id`). Без них ранние
    компоненты цепочки не попали бы в дамп вовсе: канонический `client_id`
    появляется только после IdentityResolver (§7).
    """
    cases_path = raw_meta_dir / EDGE_CASE_MANIFEST
    identity_path = raw_meta_dir / IDENTITY_MAPPING

    clean_clients = _promised_clean_clients(raw_meta_dir / DATASET_MANIFEST)

    case_clients: set[str] = set()
    if cases_path.exists():
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        for entries in cases.values():
            for entry in entries:
                client_id = entry.get("client_id")
                if client_id and client_id != "-":
                    case_clients.add(client_id)

    mapping: dict[str, dict[str, str]] = {}
    if identity_path.exists():
        mapping = json.loads(identity_path.read_text(encoding="utf-8"))

    clean: list[str] = []
    if mapping and clean_clients > 0:
        all_clients = sorted({value for section in mapping.values() for value in section.values()})
        available = [item for item in all_clients if item not in case_clients]
        if len(available) < clean_clients:
            # Молча взять сколько нашлось — тот самый случай, ради которого
            # число вынесено в манифест: фильтр отработал бы «успешно», а
            # чистого клиента в дампе не оказалось бы.
            raise DebugDumpError(
                f"набор обещает {clean_clients} клиентов без краевых случаев, "
                f"а их {len(available)}"
            )
        clean = available[:clean_clients]

    selected = case_clients | set(clean)
    if include_source_refs:
        for section in mapping.values():
            selected.update(ref for ref, client_id in section.items() if client_id in selected)

    return frozenset(selected)


def _promised_clean_clients(manifest_path: Path) -> int:
    """Сколько чистых клиентов обещает набор.

    Умолчания здесь нет намеренно. Набор без этого поля сгенерирован до того,
    как обещание появилось, и подставленное число было бы догадкой о чужих
    данных: фильтр вернул бы «чистых» клиентов, которых в наборе может не быть.
    """
    if not manifest_path.exists():
        raise DebugDumpError(
            f"нет манифеста набора {manifest_path}: сколько в нём клиентов без "
            "краевых случаев, знает только генератор"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if CLEAN_CLIENTS_KEY not in manifest:
        raise DebugDumpError(
            f"в манифесте {manifest_path} нет поля {CLEAN_CLIENTS_KEY!r} — "
            "набор сгенерирован раньше, чем появилось это обещание; перегенерируйте"
        )
    return int(manifest[CLEAN_CLIENTS_KEY])
