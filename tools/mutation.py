"""Мутационный harness — §CLAUDE.md «Мутационная проверка обязательна».

Ручной прогон мутаций дважды подвёл одинаково: мутация падала **не там**, где
задумано, а «красный прогон» принимался за доказательство. Первый раз её
перехватил Python (`non-default argument follows default argument`), второй
раз — он же, в другом файле. Записи в `CLAUDE.md` оказалось мало: её писал
тот же человек, который потом наступил снова.

Поэтому проверка перенесена из глаз в код. Мутация **объявляет**, каким
тестом и с каким сообщением она обязана быть поймана; harness сверяет это с
тем, что произошло на самом деле. Пойманная чужой ошибкой мутация получает
вердикт `WRONG_CATCH` — то есть не поймана.

Второе, что здесь закрыто, — восстановление исходника. Скрипты сравнивали
`read_text()`, а он нормализует переводы строк, поэтому «восстановлен: True»
печаталось и тогда, когда файл на диске менялся с LF на CRLF. Здесь всё
чтение и запись идут байтами, и сверка после прогона тоже байтовая.

Пример:

    from tools.mutation import Mutation, run_mutations

    run_mutations(
        [
            Mutation(
                name="сортировка убрана",
                path=SRC / "timeline_builder.py",
                replacements=(("sorted(records", "list(records"),),
                caught_by="tests/test_timeline_builder.py::test_order_is_by_timestamp",
                message="порядок событий",
            )
        ],
        tests=("tests/test_timeline_builder.py",),
    )
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Sequence

DEFAULT_LIMIT = 30.0
"""Предел на один прогон. Превышение считается пойманной мутацией не всегда:
зависание — отдельный вердикт, потому что «повис» и «упал» это разные факты."""

COLLECTION = "<collection>"
"""Мутация, которая обязана упасть при импорте, а не в конкретном тесте.

Так объявляются проверки, стоящие на уровне модуля: они срабатывают до того,
как pytest доберётся до тестов, и `FAILED <nodeid>` в выводе не появляется.
"""


class Verdict(StrEnum):
    CAUGHT = "CAUGHT"
    """Упало то, что объявлено, с объявленным сообщением."""

    WRONG_CATCH = "WRONG_CATCH"
    """Упало, но не то или не с тем сообщением. **Считается не пойманной.**"""

    SURVIVED = "SURVIVED"
    """Прогон зелёный."""

    HUNG = "HUNG"
    """Превышен предел времени."""

    NOMATCH = "NOMATCH"
    """Фрагмент для замены не найден — мутация не применилась вовсе."""


@dataclass(frozen=True)
class Mutation:
    """Одна мутация вместе с обещанием, как именно она обязана быть поймана."""

    name: str
    path: Path
    replacements: tuple[tuple[str, str], ...]

    caught_by: str
    """Ожидаемое место падения: `<файл>::<тест>` или `COLLECTION`."""

    message: str
    """Подстрока, которая обязана быть в выводе. Это и есть защита от чужой
    ошибки: у своей проверки своё сообщение, и совпасть они не могут."""

    equivalent: bool = False
    """Мутация, которая обязана **выжить**: она ничего не меняет по существу.
    Объявляется явно, чтобы «выжила» перестало быть неопределённостью."""

    reason: str = ""
    """Зачем эквивалентный мутант нужен. Без объяснения он неотличим от дыры."""


@dataclass
class Result:
    mutation: Mutation
    verdict: Verdict
    elapsed: float = 0.0
    failing: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""

    @property
    def as_expected(self) -> bool:
        if self.mutation.equivalent:
            return self.verdict is Verdict.SURVIVED
        return self.verdict is Verdict.CAUGHT


def run_mutations(
    mutations: Sequence[Mutation],
    *,
    tests: Sequence[str],
    root: Path,
    limit: float = DEFAULT_LIMIT,
    verbose: bool = True,
) -> list[Result]:
    """Прогнать мутации и сверить каждую с её обещанием.

    Возвращает результаты; ненулевой код возврата ставит вызывающий скрипт.
    Исходники читаются и пишутся байтами и восстанавливаются в `finally`
    внутри цикла: внешнее прерывание не должно оставлять мутацию в коде.
    """
    import time

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    originals = {item.path: item.path.read_bytes() for item in mutations}
    results: list[Result] = []

    for mutation in mutations:
        original = originals[mutation.path]
        text = original.decode("utf-8")

        missing = [old for old, _ in mutation.replacements if old not in text]
        if missing:
            results.append(Result(mutation, Verdict.NOMATCH, detail=missing[0][:60]))
            continue

        started = time.monotonic()
        try:
            mutated = text
            for old, new in mutation.replacements:
                mutated = mutated.replace(old, new, 1)
            mutation.path.write_bytes(mutated.encode("utf-8"))
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "pytest", *tests, "-q", "--no-header"],
                    cwd=root, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=env, stdin=subprocess.DEVNULL, timeout=limit,
                )
            except subprocess.TimeoutExpired:
                results.append(Result(mutation, Verdict.HUNG, time.monotonic() - started))
                continue
        finally:
            mutation.path.write_bytes(original)

        results.append(_judge(mutation, completed, time.monotonic() - started))

    _check_restored(originals)
    if verbose:
        _report(results)
    return results


def _judge(mutation: Mutation, completed: subprocess.CompletedProcess, elapsed: float) -> Result:
    output = (completed.stdout or "") + (completed.stderr or "")

    if completed.returncode == 0:
        return Result(mutation, Verdict.SURVIVED, elapsed)

    failing = tuple(sorted(set(re.findall(r"^FAILED ([^\s]+)", output, re.MULTILINE))))
    collected = bool(re.search(r"error[s]? during collection", output))

    if mutation.message not in output:
        return Result(
            mutation, Verdict.WRONG_CATCH, elapsed, failing,
            detail=f"нет объявленного сообщения {mutation.message!r}; "
                   f"упало: {', '.join(failing) or 'ошибка сборки'}",
        )

    if mutation.caught_by == COLLECTION:
        if not collected:
            return Result(
                mutation, Verdict.WRONG_CATCH, elapsed, failing,
                detail="объявлено падение при импорте, а упали тесты: "
                       + (", ".join(failing) or "неизвестно"),
            )
        return Result(mutation, Verdict.CAUGHT, elapsed, failing)

    if mutation.caught_by not in failing:
        return Result(
            mutation, Verdict.WRONG_CATCH, elapsed, failing,
            detail=f"объявлен {mutation.caught_by}, упали: "
                   + (", ".join(failing) or "ошибка сборки"),
        )
    return Result(mutation, Verdict.CAUGHT, elapsed, failing)


def _check_restored(originals: dict[Path, bytes]) -> None:
    """Сверка побайтная.

    Сравнение через `read_text()` здесь уже подводило: он нормализует
    переводы строк, поэтому файл, переписанный с LF на CRLF, читался как
    неизменившийся, и «восстановлен: True» печаталось при испорченном файле.
    """
    broken = sorted(str(path) for path, data in originals.items() if path.read_bytes() != data)
    if broken:
        raise RuntimeError(
            "исходники не восстановлены байт-в-байт: " + ", ".join(broken)
        )


def _report(results: Sequence[Result]) -> None:
    for item in results:
        mark = " " if item.as_expected else "  ← НЕ ТО"
        print(f"{item.verdict:<12} {item.elapsed:5.1f} c  {item.mutation.name}{mark}")
        if item.detail:
            print(f"             {item.detail}")
        elif item.verdict is Verdict.CAUGHT and item.mutation.caught_by != COLLECTION:
            print(f"             поймана: {item.mutation.caught_by}")

    unexpected = [item for item in results if not item.as_expected]
    equivalents = [item for item in results if item.mutation.equivalent]
    print(
        f"\nмутаций: {len(results)}, как объявлено: {len(results) - len(unexpected)}, "
        f"эквивалентных: {len(equivalents)}"
    )
    for item in unexpected:
        print(f"   НЕ ТО: {item.mutation.name} — {item.verdict}")
