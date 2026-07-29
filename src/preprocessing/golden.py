"""Golden-vector conformance set — §29.2.

Сравнение single/multi-worker (§29 п.10) доказывает идентичность **одной**
реализации при разном числе процессов. Между реализациями — другая машина,
другая версия библиотек, другой язык — оно не доказывает ничего. Для этого
§29.2 требует замороженный эталон: `golden_input` на входе, `golden_expected`
на выходе, побайтное совпадение как условие релиза.

**Здесь встроена вырожденность, и её надо назвать.** `golden_expected`
заполняется первым прогоном ENCODE (§29.2 п.1), то есть **генерируется из
того же кода**, с которым потом сравнивается. Первое сравнение зелёное по
построению и не доказывает ровно ничего. Доказывает только обратное: что
изменение кода делает сравнение красным. Поэтому у заморозки есть парная
проверка — прогон мутаций по каждому из четырёх артефактов §29.2 отдельно
(`scratchpad/mutate_golden.py`), и без неё «эталон совпал» читать нельзя.

**Четыре артефакта сравниваются порознь.** Один общий хэш ответил бы
«разошлось» без указания места, а места здесь разные по смыслу: события и
профиль — это данные, `bucket_field_domains` — контракт с токенайзером,
`preprocessing_state_sha256` — конфигурация. Мутация, меняющая только
события, обязана ловиться, даже когда хэш состояния не изменился.

**Заморозка требует чистого рабочего дерева** (§31, `require_clean_tree`):
эталон, снятый с кода, которого нет ни в одном коммите, воспроизвести
нельзя, а обнаружится это на чужой машине.
"""

from __future__ import annotations

import json
from dataclasses import MISSING as NO_DEFAULT
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping

from .artifacts import code_provenance, require_clean_tree
from .core.canonical import canonical_bytes, canonical_text
from .prepared_output import event_row, profile_row

COMPONENT = "golden"

# Состав §29.2 п.2, дословно.
GOLDEN_EXPECTED_ITEMS: tuple[str, ...] = (
    "prepared_profile",
    "prepared_events",
    "bucket_field_domains",
    "preprocessing_state_sha256",
)

MANIFEST_FILE = "golden_manifest.json"


class GoldenError(RuntimeError):
    """Эталон собрать или сверить нельзя — блокирующая ошибка."""


class GoldenMismatchError(RuntimeError):
    """Прогон не воспроизвёл эталон — блокирующая ошибка релиза (§29.2 п.3).

    Отдельный тип, как и у параллельности: «эталон разошёлся» и «эталон не
    снят» — разные факты, и путать их нельзя.
    """


@dataclass(frozen=True)
class GoldenExpected:
    """Четыре артефакта §29.2 п.2 — по одному полю, без умолчаний."""

    prepared_profile: Any
    prepared_events: Any
    bucket_field_domains: Any
    preprocessing_state_sha256: Any

    def document(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in dataclass_fields(self)}


def build_expected(result: Any) -> GoldenExpected:
    """Снять эталон с результата ENCODE.

    Записи берутся теми же функциями, что пишут `prepared_*` (§32): эталон
    обязан быть тем самым выходом, а не его похожей копией — иначе он
    перестанет ловить изменения формы записи.
    """
    return GoldenExpected(
        prepared_profile=[profile_row(item) for item in result.profiles()],
        prepared_events=[event_row(item) for item in result.events()],
        bucket_field_domains={
            name: list(values)
            for name, values in result.artifacts.bucket_edges.bucket_field_domains().items()
        },
        preprocessing_state_sha256=result.artifacts.versions.preprocessing_state_sha256,
    )


def freeze_expected(
    expected: GoldenExpected,
    target: Path,
    *,
    root: Path,
    golden_vectors_version: str,
    dataset_identifier: str,
) -> list[Path]:
    """Записать эталон (§29.2 п.1) и манифест, чем он снят.

    Чистое дерево обязательно: §29.2 п.3 делает несовпадение блокирующей
    ошибкой релиза, а эталон с неизвестного кода превращает эту ошибку в
    неразрешимую.
    """
    provenance = code_provenance(root)
    require_clean_tree(provenance, "заморозка golden_expected")

    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in sorted(expected.document().items()):
        path = target / f"{name}.json"
        path.write_bytes(canonical_bytes(payload))
        written.append(path)

    manifest = {
        "golden_vectors_version": golden_vectors_version,
        "dataset_identifier": dataset_identifier,
        "items": list(GOLDEN_EXPECTED_ITEMS),
        # Чем снят эталон: без этого «не совпало» не с чем сопоставить.
        "code_state_sha256": provenance["code_state_sha256"],
        "git_commit": provenance["git_commit"],
        "python": _python_version(),
    }
    path = target / MANIFEST_FILE
    path.write_bytes(canonical_bytes(manifest))
    written.append(path)
    return written


def load_expected(target: Path) -> GoldenExpected:
    """Прочитать замороженный эталон."""
    values: dict[str, Any] = {}
    for name in GOLDEN_EXPECTED_ITEMS:
        path = target / f"{name}.json"
        if not path.exists():
            raise GoldenError(f"в эталоне нет {name}.json — набор §29.2 неполон")
        values[name] = json.loads(path.read_text(encoding="utf-8"))
    return GoldenExpected(**values)


def compare_expected(actual: GoldenExpected, expected: GoldenExpected) -> None:
    """Сверить прогон с эталоном по каждому артефакту отдельно (§29.2 п.2).

    Порознь, а не одним хэшем: общий хэш сказал бы «разошлось» без указания
    места, а места разные по смыслу — данные, контракт с токенайзером,
    конфигурация. Расхождение по одному артефакту при совпадении остальных —
    самый частый и самый информативный случай.
    """
    differences: list[str] = []
    for name in GOLDEN_EXPECTED_ITEMS:
        mine = canonical_bytes(getattr(actual, name))
        theirs = canonical_bytes(getattr(expected, name))
        if mine != theirs:
            differences.append(f"{name}: {_describe(name, getattr(actual, name), getattr(expected, name))}")

    if differences:
        raise GoldenMismatchError(
            "прогон не воспроизвёл golden_expected (§29.2 п.3, блокирующая ошибка релиза): "
            + "; ".join(differences)
        )


def _describe(name: str, actual: Any, expected: Any) -> str:
    """Короткое указание места расхождения.

    Печатать целиком 374 события бессмысленно — нужен первый различающийся
    элемент, по нему и ищут.
    """
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"записей {len(actual)} против {len(expected)}"
        for index, (mine, theirs) in enumerate(zip(actual, expected)):
            if canonical_bytes(mine) != canonical_bytes(theirs):
                return f"первое расхождение в записи {index}: {_first_field(mine, theirs)}"
        return "длины равны, содержимое совпало — расхождение в порядке сериализации"

    if isinstance(actual, dict) and isinstance(expected, dict):
        keys = sorted(set(actual) | set(expected))
        for key in keys:
            if actual.get(key) != expected.get(key):
                return f"ключ {key!r}: {actual.get(key)!r} против {expected.get(key)!r}"

    return f"{canonical_text(actual)[:80]} против {canonical_text(expected)[:80]}"


def _first_field(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    for key in sorted(set(actual) | set(expected)):
        if actual.get(key) != expected.get(key):
            return f"поле {key!r}: {actual.get(key)!r} против {expected.get(key)!r}"
    return "поля совпали"


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _check_expected_is_complete() -> None:
    """Состав §29.2 п.2 совпадает с полями, и ни у одного нет умолчания."""
    declared = {item.name for item in dataclass_fields(GoldenExpected)}

    missing = sorted(set(GOLDEN_EXPECTED_ITEMS) - declared)
    if missing:
        raise RuntimeError("в эталоне нет артефактов §29.2: " + ", ".join(missing))

    extra = sorted(declared - set(GOLDEN_EXPECTED_ITEMS))
    if extra:
        raise RuntimeError("в эталоне есть артефакты сверх §29.2: " + ", ".join(extra))

    optional = sorted(
        item.name
        for item in dataclass_fields(GoldenExpected)
        if item.default is not NO_DEFAULT or item.default_factory is not NO_DEFAULT
    )
    if optional:
        raise RuntimeError(
            "артефакты эталона со значением по умолчанию: "
            + ", ".join(optional)
            + " — такой артефакт можно не заполнить, и эталон замёрзнет неполным"
        )


_check_expected_is_complete()
