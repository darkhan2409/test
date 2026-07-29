"""Детерминированный порядок событий — §13, §26.

Самая важная сортировка в проекте: порядок событий и есть то, что читает
модель. Поедет он — поедет всё, и ни один счётчик этого не заметит: записи
останутся на месте, изменится только их последовательность.

**Про вырожденные тесты.** Три раза за проект проверка порядка оказывалась
пустой, и каждый раз вырожденность была своя:

1. партиции — файловая система отдавала файлы уже отсортированными;
2. экраны — `SCREEN_01..12`, алфавит совпадал с хронологией;
3. здесь — `source_record_id` строился из того же индекса, что и время,
   поэтому третий ключ сортировки в одиночку воспроизводил порядок первого,
   и код без `timestamp_utc` в ключе проходил все тесты.

Отсюда две страховки, и они сторожат **разные** свойства:

- `assert_input_is_not_ordered` — вход не отсортирован заранее. Ловит
  «сортировки нет вовсе»;
- `assert_key_is_decisive` — без указанной части ключа порядок обязан
  измениться. Ловит «часть ключа выпала». Первая страховка здесь честно
  проходила и не помогала: вход был перемешан, вырожденность сидела в том,
  что два разных ключа давали один ответ.

Ни одна из них не универсальна. Что данные не выродились по третьему,
четвёртому и любому следующему признаку, показывает только мутационная
проверка; страховки нужны, чтобы уже найденная вырожденность не вернулась
молча.

Оба tie-break §13 покрыты раздельно: одинаковый момент в разных источниках
(решает `source_priority`) и в одном (решает `source_record_id`).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.preprocessing.feature_projection import PROFILE_SECTION, ProjectedRecord
from src.preprocessing.schema import load_source_contracts
from src.preprocessing.timeline_builder import TimelineBuilder, TimelineError

UTC = timezone.utc
CONFIG = Path("config")
CUTOFF = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)
START = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)

# Части ключа сортировки §13 — по позиции в кортеже `TimelineBuilder._sort_key`.
KEY_TIMESTAMP, KEY_PRIORITY, KEY_RECORD_ID = 0, 1, 2
KEY_NAMES = {
    KEY_TIMESTAMP: "timestamp_utc",
    KEY_PRIORITY: "source_priority",
    KEY_RECORD_ID: "source_record_id",
}


def anti_chronological_id(minutes: int) -> str:
    """Идентификатор, который по алфавиту идёт ПРОТИВ времени.

    Значение по умолчанию для `event()`: чем позже событие, тем «меньше» его
    `source_record_id` как строка. Сортировка по одному идентификатору даёт
    порядок, обратный правильному, — поэтому выпадение `timestamp_utc` из
    ключа больше не может остаться незамеченным.

    Совпадение алфавита с хронологией здесь не «маловероятно», а невозможно
    по построению: именно им тест и был обесценен.
    """
    return f"CP-{1000 - minutes:04d}"


@pytest.fixture(name="registry")
def registry_fixture():
    return load_source_contracts(CONFIG / "source_contracts.yaml")


def event(
    *,
    minutes: int,
    source: str = "core_payments",
    record_id: str | None = None,
    client_id: str = "C000001",
    event_type: str = "TRANSFER",
) -> ProjectedRecord:
    return ProjectedRecord(
        source=source,
        partition=f"{source}/2026-01-01.jsonl",
        line_number=minutes,
        source_record_id=record_id if record_id is not None else anti_chronological_id(minutes),
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id=client_id,
        timestamp_utc=START + timedelta(minutes=minutes),
        calendar_timezone="Asia/Almaty",
        event_type=event_type,
        event_id=f"{minutes:032x}",
        fields={},
        schema_section=event_type,
    )


def build(registry, records) -> list[ProjectedRecord]:
    builder = TimelineBuilder(registry, cutoff=CUTOFF)
    return [r for r in builder.build(records) if r.event_type is not None]


# --------------------------------------------------------------------------- #
# Страховки от вырожденных данных
# --------------------------------------------------------------------------- #


def _full_key(record, priority) -> tuple:
    return (record.timestamp_utc, priority[record.source], record.source_record_id)


def _order_without(records, registry, dropped: int) -> list[str]:
    """Порядок, который получился бы БЕЗ одной части ключа.

    Ровно то, что делает мутация «убран N-й ключ». `sorted` устойчив, поэтому
    когда все различающие части выпали, остаётся порядок поступления — то
    есть нарезка потока по воркерам, от которой §13 и защищает.
    """
    priority = registry.source_priority()
    return [
        record.source_record_id
        for record in sorted(
            records,
            key=lambda r: tuple(
                value for index, value in enumerate(_full_key(r, priority)) if index != dropped
            ),
        )
    ]


def assert_input_is_not_ordered(records, registry) -> None:
    """Страховка №1: вход обязан быть неотсортированным.

    Без неё «проверка порядка» превращается в проверку того, что сортировка
    не портит уже отсортированное, — а это проходит и без самой сортировки.
    """
    priority = registry.source_priority()
    keys = [_full_key(record, priority) for record in records]
    assert keys != sorted(keys), "вход уже отсортирован — тест ничего не проверяет"


def assert_key_is_decisive(records, registry, dropped: int) -> None:
    """Страховка №2: без этой части ключа порядок обязан измениться.

    Проверяет свойство самих данных, а не кода: если убрать часть ключа и
    результат тот же, значит на этом наборе части ключа коллинеарны и тест
    не отличит код с ней от кода без неё. Так и выжила мутация «убран
    `timestamp_utc`»: идентификаторы шли по возрастанию времени.
    """
    priority = registry.source_priority()
    full = [record.source_record_id for record in sorted(records, key=lambda r: _full_key(r, priority))]
    assert _order_without(records, registry, dropped) != full, (
        f"без {KEY_NAMES[dropped]} порядок тот же — на этих данных тест не отличит "
        f"код с этой частью ключа от кода без неё"
    )


# --------------------------------------------------------------------------- #
# Основной порядок
# --------------------------------------------------------------------------- #


def test_reversed_input_is_ordered_by_timestamp(registry):
    """Обратный порядок на входе — прямой на выходе."""
    records = [event(minutes=index) for index in range(10)]
    shuffled = list(reversed(records))
    assert_input_is_not_ordered(shuffled, registry)
    assert_key_is_decisive(shuffled, registry, KEY_TIMESTAMP)

    ordered = build(registry, shuffled)

    assert [r.source_record_id for r in ordered] == [
        anti_chronological_id(index) for index in range(10)
    ]
    assert [r.position for r in ordered] == list(range(10))


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_shuffled_input_gives_the_same_timeline(registry, seed):
    """Перемешанный вход даёт тот же timeline при любом перемешивании."""
    records = [event(minutes=index) for index in range(20)]
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    assert_input_is_not_ordered(shuffled, registry)
    assert_key_is_decisive(shuffled, registry, KEY_TIMESTAMP)

    assert [r.ordering_key for r in build(registry, shuffled)] == [
        r.ordering_key for r in build(registry, records)
    ]


def test_ordering_key_is_monotonic(registry):
    """`ordering_key` растёт вдоль timeline — им можно сортировать заново.

    Ключ уезжает в выходные данные (§5 п.7), и читатель вправе полагаться
    на то, что строковое сравнение даёт тот же порядок.
    """
    records = [event(minutes=index) for index in range(15)]
    shuffled = list(records)
    random.Random(5).shuffle(shuffled)
    assert_input_is_not_ordered(shuffled, registry)
    assert_key_is_decisive(shuffled, registry, KEY_TIMESTAMP)

    keys = [r.ordering_key for r in build(registry, shuffled)]

    assert keys == sorted(keys)


def test_priority_is_zero_padded_in_the_key(registry):
    """Ранг печатается фиксированной шириной.

    Без ведущих нулей строковое сравнение поставило бы источник с рангом `10`
    раньше источника с рангом `9` — то есть `ordering_key` перестал бы
    совпадать с настоящим порядком.
    """
    builder = TimelineBuilder(registry, cutoff=CUTOFF)
    key = builder.ordering_key(event(minutes=0, record_id="CP-1"))

    moment, priority, record_id = key.split("|")
    assert priority == "000010"  # core_payments объявлен рангом 10
    assert record_id == "CP-1"
    assert moment.endswith("Z")


def test_every_part_of_the_key_decides_something(registry):
    """Набор, на котором значима каждая из трёх частей ключа §13.

    Обобщение находки: коллинеарность может связать любую пару частей ключа,
    не только ту, что попалась. Здесь данные подобраны так, что выпадение
    **любой** части меняет результат, и это утверждается явно — тремя
    страховками — до самой проверки порядка.
    """
    records = [
        event(minutes=0, record_id="R9"),                          # ранг 10
        event(minutes=0, source="app_logs", record_id="R1",        # ранг 30
              event_type="PUSH_NOTIFICATION"),
        event(minutes=0, record_id="R5"),                          # ранг 10
        event(minutes=5, record_id="R0"),                          # позже всех, id меньше всех
    ]
    assert_input_is_not_ordered(records, registry)
    for part in (KEY_TIMESTAMP, KEY_PRIORITY, KEY_RECORD_ID):
        assert_key_is_decisive(records, registry, part)

    ordered = build(registry, records)

    assert [r.source_record_id for r in ordered] == ["R5", "R9", "R1", "R0"]


# --------------------------------------------------------------------------- #
# Tie-break 1: разные источники, один момент → source_priority
# --------------------------------------------------------------------------- #


def test_tie_between_sources_is_resolved_by_source_priority(registry):
    """Одинаковый момент в разных системах решает ранг источника (§13 п.2).

    Записи поданы в порядке, обратном рангу, и `source_record_id` у них
    подобран так, чтобы третий ключ дал противоположный ответ: если ранг
    потеряется, порядок окажется другим, а не случайно тем же.
    """
    # core_payments = 10, card_processing = 20, app_logs = 30.
    later_rank = event(minutes=0, source="app_logs", record_id="AAA-1",
                       event_type="PUSH_NOTIFICATION")
    middle_rank = event(minutes=0, source="card_processing", record_id="BBB-1",
                        event_type="CARD_PURCHASE")
    first_rank = event(minutes=0, source="core_payments", record_id="ZZZ-1")

    records = [later_rank, middle_rank, first_rank]
    assert_input_is_not_ordered(records, registry)
    assert_key_is_decisive(records, registry, KEY_PRIORITY)

    ordered = build(registry, records)

    assert [r.source for r in ordered] == ["core_payments", "card_processing", "app_logs"]


def test_source_priority_beats_record_id(registry):
    """Ранг сильнее третьего ключа: порядок задаёт владелец данных.

    У события старшего источника идентификатор заведомо «больше» по строке.
    Если бы решал `source_record_id`, оно уехало бы в конец.
    """
    high_rank = event(minutes=0, source="core_payments", record_id="ZZZZ")
    low_rank = event(minutes=0, source="app_logs", record_id="AAAA",
                     event_type="PUSH_NOTIFICATION")

    records = [low_rank, high_rank]
    assert_input_is_not_ordered(records, registry)
    assert_key_is_decisive(records, registry, KEY_PRIORITY)

    ordered = build(registry, records)

    assert [r.source_record_id for r in ordered] == ["ZZZZ", "AAAA"]


# --------------------------------------------------------------------------- #
# Tie-break 2: один источник, один момент → source_record_id
# --------------------------------------------------------------------------- #


def test_tie_within_one_source_is_resolved_by_record_id(registry):
    """Одинаковый момент в одном источнике решает `source_record_id` (§13 п.3).

    Здесь рангу решать нечего — он совпадает, — и без третьего ключа порядок
    определяла бы очередь поступления, то есть нарезка потока по воркерам.
    """
    records = [
        event(minutes=0, record_id="CP-tie-c"),
        event(minutes=0, record_id="CP-tie-a"),
        event(minutes=0, record_id="CP-tie-b"),
    ]
    assert_input_is_not_ordered(records, registry)
    assert_key_is_decisive(records, registry, KEY_RECORD_ID)

    ordered = build(registry, records)

    assert [r.source_record_id for r in ordered] == ["CP-tie-a", "CP-tie-b", "CP-tie-c"]


def test_same_source_tie_is_stable_under_permutation(registry):
    """Порядок при совпадении момента не зависит от порядка поступления."""
    records = [event(minutes=0, record_id=f"CP-{letter}") for letter in "fedcba"]

    results = set()
    for seed in range(6):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        results.add(tuple(r.source_record_id for r in build(registry, shuffled)))

    assert len(results) == 1
    assert results.pop() == tuple(f"CP-{letter}" for letter in "abcdef")


def test_report_shows_which_key_resolved_the_tie(registry):
    """Отчёт различает оба tie-break.

    Нулевые счётчики на данных, где совпадения заведомо есть, означают, что
    проверка порядка ничего не проверяет.
    """
    builder = TimelineBuilder(registry, cutoff=CUTOFF)
    list(builder.build([
        event(minutes=0, source="core_payments", record_id="CP-1"),
        event(minutes=0, source="card_processing", record_id="CRD-1",
              event_type="CARD_PURCHASE"),
        event(minutes=0, source="card_processing", record_id="CRD-2",
              event_type="CARD_PURCHASE"),
    ]))
    summary = builder.report.summary()

    assert summary["ties_resolved_by_source_priority"] == 1
    assert summary["ties_resolved_by_source_record_id"] == 1


# --------------------------------------------------------------------------- #
# Границы компонента
# --------------------------------------------------------------------------- #


def test_clients_do_not_mix(registry):
    """Timeline строится по клиенту: чужие события не попадают в чужой порядок."""
    records = [
        event(minutes=5, client_id="C000002"),
        event(minutes=1, client_id="C000001"),
        event(minutes=9, client_id="C000001"),
    ]

    ordered = build(registry, records)
    by_client: dict[str, list[str]] = {}
    for record in ordered:
        by_client.setdefault(record.client_id, []).append(record.source_record_id)

    assert by_client == {
        "C000001": [anti_chronological_id(1), anti_chronological_id(9)],
        "C000002": [anti_chronological_id(5)],
    }
    assert [r.position for r in ordered if r.client_id == "C000001"] == [0, 1]
    assert [r.position for r in ordered if r.client_id == "C000002"] == [0]


def test_event_after_cutoff_is_blocking(registry):
    """§26 п.5: событие позже T в timeline не попадает.

    Сюда оно может прийти только мимо §14, и это ошибка кода, а не данных.
    """
    late = event(minutes=0)
    leaked = type(late)(**{**late.__dict__, "timestamp_utc": CUTOFF + timedelta(seconds=1)})

    with pytest.raises(TimelineError, match="позже T"):
        build(registry, [leaked])


def test_event_without_record_id_is_blocking(registry):
    """Без третьего ключа tie-break §13 нечем завершить."""
    broken = event(minutes=0)
    empty = type(broken)(**{**broken.__dict__, "source_record_id": ""})

    with pytest.raises(TimelineError, match="третий ключ"):
        build(registry, [empty])


def test_profile_passes_through_untouched(registry):
    """Профиль привязан к T целиком и в timeline не сортируется."""
    profile = ProjectedRecord(
        source="profile_snapshots", partition="p", line_number=1,
        source_record_id="CIF-1", source_schema_version="1.0", client_ref="CIF000001",
        payload={}, client_id="C000001", timestamp_utc=None, calendar_timezone="Asia/Almaty",
        event_type=None, event_id=None, fields={}, schema_section=PROFILE_SECTION,
    )
    builder = TimelineBuilder(registry, cutoff=CUTOFF)

    result = list(builder.build([profile, event(minutes=0)]))

    assert result[0] is profile
    assert builder.report.summary()["events"] == 1


def test_ordering_key_still_carries_the_source_record_id(registry):
    """Сторож на охраняемую зависимость: lineage §8 держится на составе ключа.

    Из `prepared_events` вернуться к сырой записи можно только через
    `ordering_key`: `source_meta` из выхода убран сознательно (§32.2, §2.2),
    и `source_record_id` доезжает до токенайзера **внутри ключа** — как третья
    часть tie-break §13, а не как lineage.

    Совпадение это или замысел, но пока оно есть, от него зависит §8. §13
    при этом прямо разрешает третьим ключом `event_id` — такая правка молча
    оборвала бы единственный путь к сырой записи, не нарушив ни одного
    правила. Тест превращает совпадение в зависимость, о которой сообщают.
    """
    builder = TimelineBuilder(registry, cutoff=CUTOFF)
    record = event(minutes=0, record_id="CP-XYZ-001")

    key = builder.ordering_key(record)

    assert record.source_record_id in key, (
        "ordering_key перестал нести source_record_id — от этого зависит "
        "lineage §8: из prepared_events это единственный путь к сырой записи. "
        "Меняешь третий ключ tie-break §13 — обеспечь lineage иначе"
    )
