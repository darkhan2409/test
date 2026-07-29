"""Локальные календарные признаки — §25, §25.1, §25.3.

Проверяется то, что нельзя увидеть глазами и что разъезжается молча:

- час и день считаются в бизнес-зоне, а не из UTC (§25, QA §35 п.8);
- до 2024-03-01 берутся исторические правила IANA, а не нынешнее смещение
  (§12.1, QA §35 пп. 9–10) — на Windows это ещё и проверка, что пакет
  `tzdata` действительно отдаёт историю;
- конвенция понедельник = 0 (§5 п.12): без неё golden-vectors не сойдутся,
  а расхождение видно только по номеру;
- `lifetime_first` стоит ровно на самом раннем событии клиента (§25.1).

**Про вырожденные тесты.** Проверка «час считается по зоне» пуста, если
момент выбран так, что локальный час совпадает с часом UTC. Поэтому каждый
такой тест сначала утверждает, что данные различают правильный ответ и
неправильный, и падает, если однажды перестанут.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.preprocessing.feature_projection import PROFILE_SECTION, ProjectedRecord
from src.preprocessing.time_feature_builder import TimeFeatureBuilder, TimeFeatureError
from src.preprocessing.timeline_builder import TimelineRecord

UTC = timezone.utc
ALMATY = "Asia/Almaty"

# Смена времени в Казахстане: до 2024-03-01 UTC+6, после — UTC+5 (§12.1).
BEFORE_SWITCH = datetime(2024, 2, 28, 20, 0, tzinfo=UTC)
AFTER_SWITCH = datetime(2024, 3, 5, 20, 0, tzinfo=UTC)

# 2026-01-05 — понедельник, 2026-01-11 — воскресенье.
MONDAY = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
SUNDAY = datetime(2026, 1, 11, 12, 0, tzinfo=UTC)


def event(
    *,
    moment: datetime,
    position: int = 0,
    client_id: str = "C000001",
    timezone_name: str | None = ALMATY,
    event_id: str = "e" * 32,
) -> TimelineRecord:
    return TimelineRecord(
        source="core_payments",
        partition="core_payments/2026-01-01.jsonl",
        line_number=position,
        source_record_id=f"CP-{position:03d}",
        source_schema_version="1.0",
        client_ref="000001",
        payload={},
        client_id=client_id,
        timestamp_utc=moment,
        calendar_timezone=timezone_name,
        event_type="TRANSFER",
        event_id=event_id,
        fields={},
        schema_section="TRANSFER",
        ordering_key=f"{moment.isoformat()}|000010|CP-{position:03d}",
        position=position,
    )


def build(records) -> list:
    builder = TimeFeatureBuilder()
    return list(builder.build(records))


# --------------------------------------------------------------------------- #
# Страховки от вырожденных данных
# --------------------------------------------------------------------------- #


def assert_local_hour_differs_from_utc(moment: datetime) -> None:
    """Момент обязан различать «по зоне» и «по UTC».

    Если локальный час совпал с часом UTC, тест пройдёт и на коде, который
    зону игнорирует, — то есть не проверит ровно то, ради чего написан.
    """
    local = moment.astimezone(ZoneInfo(ALMATY))
    assert local.hour != moment.hour, (
        f"{moment.isoformat()}: локальный час совпал с UTC — тест не отличит "
        f"расчёт по зоне от расчёта по UTC"
    )


def assert_historical_rules_are_available() -> None:
    """У зоны обязана быть история, иначе проверка §12.1 пуста.

    На Windows исторические правила приходят пакетом `tzdata`. Без него
    `ZoneInfo` отдаст одно и то же смещение на обе даты, и тест «до и после
    перехода» пройдёт, ничего не проверив.
    """
    zone = ZoneInfo(ALMATY)
    before = BEFORE_SWITCH.astimezone(zone).utcoffset()
    after = AFTER_SWITCH.astimezone(zone).utcoffset()
    assert before - after == timedelta(hours=1), (
        f"смещения до и после 2024-03-01 совпали ({before} и {after}) — "
        f"исторических правил IANA нет, проверять нечего"
    )


# --------------------------------------------------------------------------- #
# Локальная зона, а не UTC
# --------------------------------------------------------------------------- #


def test_hour_is_local_not_utc():
    """§25, QA §35 п.8: час считается в бизнес-зоне."""
    moment = datetime(2026, 1, 15, 18, 30, tzinfo=UTC)  # 23:30 в Алматы
    assert_local_hour_differs_from_utc(moment)

    features = build([event(moment=moment)])[0].calendar_time_features

    assert features.hour_of_day_local == 23
    assert moment.hour == 18  # именно то значение, которое дал бы расчёт из UTC


def test_history_before_2024_uses_historical_rules():
    """§12.1, QA §35 пп. 9–10: до 2024-03-01 Алматы жил в UTC+6.

    Ретроактивное применение нынешнего UTC+05 сдвинуло бы всю историю на час
    и не оставило следа — событие просто оказалось бы «часом раньше».
    """
    assert_historical_rules_are_available()

    old, new = build([
        event(moment=BEFORE_SWITCH, position=0),
        event(moment=AFTER_SWITCH, position=1),
    ])

    # 20:00 UTC → 02:00 следующего дня при +6 и 01:00 при +5.
    assert old.calendar_time_features.hour_of_day_local == 2
    assert new.calendar_time_features.hour_of_day_local == 1


def test_missing_timezone_is_blocking_not_utc():
    """Без зоны обработка останавливается, а не считает по UTC.

    Подстановка UTC дала бы правдоподобное число вместо ошибки: признак
    «ночная активность» тихо превратился бы в «вечернюю».
    """
    with pytest.raises(TimeFeatureError, match="из UTC нельзя"):
        build([event(moment=MONDAY, timezone_name=None)])


def test_unknown_timezone_is_blocking():
    """Неизвестная зона — ошибка кода: §12 обязан был отсечь запись раньше."""
    with pytest.raises(TimeFeatureError, match="не является IANA timezone"):
        build([event(moment=MONDAY, timezone_name="Asia/Nowhere")])


# --------------------------------------------------------------------------- #
# Конвенция дня недели
# --------------------------------------------------------------------------- #


def test_monday_is_zero_and_sunday_is_six():
    """§5 п.12: понедельник = 0, воскресенье = 6.

    Две другие ходовые конвенции дали бы здесь другие числа: при «Пн = 1»
    понедельник стал бы единицей, при «Вс = 0» воскресенье — нулём.
    Обе отсекаются этими двумя значениями.
    """
    assert MONDAY.isoweekday() == 1, "выбранная дата не понедельник"
    assert SUNDAY.isoweekday() == 7, "выбранная дата не воскресенье"

    monday, sunday = build([
        event(moment=MONDAY, position=0),
        event(moment=SUNDAY, position=1),
    ])

    assert monday.calendar_time_features.day_of_week_local == 0
    assert sunday.calendar_time_features.day_of_week_local == 6


def test_week_covers_zero_to_six_exactly_once():
    """Семь подряд идущих дней дают все значения 0..6 по одному разу.

    Страховка от вырожденности самой конвенции: сдвиг или заворот проявятся
    повтором или пропуском, а не «похожими» числами.
    """
    records = [
        event(moment=MONDAY + timedelta(days=offset), position=offset)
        for offset in range(7)
    ]

    days = [r.calendar_time_features.day_of_week_local for r in build(records)]

    assert days == [0, 1, 2, 3, 4, 5, 6]


def test_day_of_week_is_local_too():
    """День недели тоже локальный: у ночного события UTC отстаёт на сутки."""
    moment = datetime(2026, 1, 11, 20, 0, tzinfo=UTC)  # вс 20:00 UTC → пн 01:00
    assert moment.isoweekday() == 7

    record = build([event(moment=moment)])[0]

    assert record.calendar_time_features.day_of_week_local == 0  # уже понедельник
    assert moment.weekday() == 6  # значение, которое дал бы расчёт из UTC


# --------------------------------------------------------------------------- #
# lifetime_first — §5 п.13, §25.1
# --------------------------------------------------------------------------- #


def test_lifetime_first_only_on_the_earliest_event_of_each_client():
    """Флаг стоит ровно один раз на клиента и именно на его первом событии."""
    start = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
    records = [
        event(moment=start + timedelta(hours=index), position=index, client_id="C000001")
        for index in range(3)
    ] + [
        event(moment=start + timedelta(hours=index), position=index, client_id="C000002")
        for index in range(2)
    ]

    result = build(records)
    flagged = [r for r in result if r.lifetime_first]

    assert len(flagged) == 2, "флаг обязан быть ровно у одного события каждого клиента"
    assert {r.client_id for r in flagged} == {"C000001", "C000002"}
    for record in flagged:
        earliest = min(
            (r for r in result if r.client_id == record.client_id),
            key=lambda r: r.timestamp_utc,
        )
        assert record is earliest


def test_truncated_timeline_is_blocking():
    """Обрезанная история — блокирующая ошибка, а не тихо ложный флаг.

    Если timeline обрезать до §25, `position == 0` перестанет означать начало
    жизни клиента и станет началом окна. Токенайзер (§25.1) различает
    `FIRST_EVENT` и `WINDOW_START` именно этим флагом, и соврал бы он молча.
    """
    with pytest.raises(TimeFeatureError, match="timeline обрезан"):
        build([event(moment=MONDAY, position=3)])


def test_event_without_position_is_blocking():
    """Без §13 порядок неизвестен, и выводить lifetime_first не из чего."""
    without_order = ProjectedRecord(
        source="core_payments", partition="p", line_number=1,
        source_record_id="CP-1", source_schema_version="1.0", client_ref="000001",
        payload={}, client_id="C000001", timestamp_utc=MONDAY,
        calendar_timezone=ALMATY, event_type="TRANSFER", event_id="e" * 32,
        fields={}, schema_section="TRANSFER",
    )

    with pytest.raises(TimeFeatureError, match="без position"):
        build([without_order])


# --------------------------------------------------------------------------- #
# Границы компонента
# --------------------------------------------------------------------------- #


def test_profile_passes_through_untouched():
    """У профиля локального часа нет: он привязан к T целиком (§6)."""
    profile = ProjectedRecord(
        source="profile_snapshots", partition="p", line_number=1,
        source_record_id="CIF-1", source_schema_version="1.0", client_ref="CIF000001",
        payload={}, client_id="C000001", timestamp_utc=None, calendar_timezone=ALMATY,
        event_type=None, event_id=None, fields={}, schema_section=PROFILE_SECTION,
    )
    builder = TimeFeatureBuilder()

    result = list(builder.build([profile, event(moment=MONDAY)]))

    assert result[0] is profile
    assert builder.report.summary()["events"] == 1


def test_nothing_from_the_timeline_record_is_lost():
    """Достройка признаками не теряет уже накопленного.

    Компонент собирает новый тип записи, а не правит старую. Потерянный при
    сборке атрибут — `position`, `ordering_key`, `quality_flags` — не поднимет
    ошибки: запись доедет до выдачи обеднённой и молча. Проверяются все поля
    родителя разом, потому что список полей будет расти, а тест — нет.
    """
    source = event(moment=MONDAY)

    result = build([source])[0]

    for field in dataclass_fields(TimelineRecord):
        assert getattr(result, field.name) == getattr(source, field.name), (
            f"поле {field.name} потерялось при достройке календарных признаков"
        )


def test_report_counts_the_shift_from_utc():
    """Счётчик сдвига — детектор расчёта из UTC, а не статистика.

    На зоне, отличной от UTC, он обязан быть ненулевым. Ноль означает, что
    зона не применяется — то самое нарушение §25, которое иначе не видно.
    """
    builder = TimeFeatureBuilder()
    list(builder.build([
        event(moment=MONDAY, position=0),
        event(moment=SUNDAY, position=1),
    ]))
    summary = builder.report.summary()

    assert summary["events"] == 2
    assert summary["hours_shifted_from_utc"] == 2
    assert summary["lifetime_first"] == 1
