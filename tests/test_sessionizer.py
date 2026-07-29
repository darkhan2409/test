"""Sessionizer — §20, и его стык с маппингом событий §10.

Две проверки, обе про тихие ошибки.

**Согласованность двух конфигов.** `event_mapping.yaml` помечает действия
app-лога как «не событие» (§10 п.5), `sessionization.yaml` собирает их в
`APP_SESSION`. Это одно и то же множество, описанное с двух сторон. Разойдись
они — действие не станет ни событием, ни частью сессии: оно просто исчезнет,
не уронив прогон и не попав в карантин. Тот же класс, что рассогласование
границы `T` в §12/§14: два места реализуют одно правило по-разному.

**`screens_count` против длины `screens`.** Первое считает все уникальные
экраны сессии, второе обрезано до `max_values_per_field` (§20.3, §21). Числа
разные намеренно, и это выглядит как баг — ровно поэтому здесь тест: «починка»
из лучших побуждений заставит §19 бакетизировать другую величину, и
`screens_count_bucket` станет означать не то, что означал.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.event_mapper import EventMapping, load_event_mapping
from src.preprocessing.event_mapper import MappedRecord
from src.preprocessing.feature_projection import FeatureProjector, ProjectedRecord, load_feature_schema
from src.preprocessing.field_policies import FieldPolicies
from src.preprocessing.schema import load_source_contracts
from src.preprocessing.sessionizer import (
    SESSION_EVENT_TYPE,
    SessionizationError,
    Sessionizer,
    load_sessionization_config,
)

UTC = timezone.utc
CONFIG = Path("config")
SESSION_GAP = timedelta(minutes=30)
MAX_VALUES = 8
START = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)


@pytest.fixture(name="configs")
def configs_fixture():
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    mapping = load_event_mapping(CONFIG / "event_mapping.yaml", registry)
    return registry, mapping


def remap_app_logs(mapping: EventMapping, codes: dict[str, str | None]) -> EventMapping:
    """Собрать маппинг с другим набором кодов app-логов."""
    document = mapping.model_dump(mode="json")
    document["sources"]["app_logs"] = {
        "code_field": "action",
        "codes": codes,
        "default_event_type": None,
    }
    return EventMapping.model_validate(document)


# --------------------------------------------------------------------------- #
# Согласованность конфигов
# --------------------------------------------------------------------------- #


def test_configs_agree_on_which_actions_build_sessions(configs):
    """Настоящие конфиги согласованы — базовая точка отсчёта для мутаций."""
    registry, mapping = configs
    config = load_sessionization_config(CONFIG / "sessionization.yaml", registry, mapping)

    technical = {
        code for code, event_type in mapping.sources[config.source].codes.items()
        if event_type is None
    }
    assert technical == set(config.session_actions)


BASE_CODES = {
    "push_received": "PUSH_NOTIFICATION",
    "LOGIN": None,
    "screen_open": None,
    "button_click": None,
    "form_submit": None,
}


def test_new_technical_action_not_declared_in_sessions_is_blocking(configs):
    """Код без типа события, о котором сессионизация не знает, исчез бы молча.

    Самый опасный из трёх: событием строка не станет, в сессию не попадёт,
    в карантин не уедет. Ни счётчика, ни ошибки — просто на несколько строк
    меньше.
    """
    registry, mapping = configs
    drifted = remap_app_logs(mapping, {**BASE_CODES, "widget_tap": None})

    with pytest.raises(SessionizationError, match="исчезнут молча"):
        load_sessionization_config(CONFIG / "sessionization.yaml", registry, drifted)


def test_action_promoted_to_event_is_blocking(configs):
    """Действие, ставшее событием в маппинге, но оставшееся в сессии.

    Одна и та же строка попала бы в timeline дважды: как самостоятельное
    событие и внутри `APP_SESSION`.
    """
    registry, mapping = configs
    drifted = remap_app_logs(mapping, {**BASE_CODES, "form_submit": "PUSH_NOTIFICATION"})

    with pytest.raises(SessionizationError, match="попадёт в timeline дважды"):
        load_sessionization_config(CONFIG / "sessionization.yaml", registry, drifted)


def test_action_missing_from_mapping_is_blocking(configs):
    """Действие сессии, которого в маппинге нет вовсе.

    Единственный из трёх случаев, который виден снаружи: EventMapper отправит
    строки в карантин как неизвестный код. Всё равно блокирующий — сессии
    молча похудеют.
    """
    registry, mapping = configs
    shrunk = remap_app_logs(
        mapping, {code: value for code, value in BASE_CODES.items() if code != "form_submit"}
    )

    with pytest.raises(SessionizationError, match="в маппинге их нет вовсе"):
        load_sessionization_config(CONFIG / "sessionization.yaml", registry, shrunk)


# --------------------------------------------------------------------------- #
# screens_count против screens
# --------------------------------------------------------------------------- #


def action(index: int, screen: str | None, *, act: str = "screen_open") -> MappedRecord:
    return MappedRecord(
        source="app_logs",
        partition="app_logs/2026-01-01.jsonl",
        line_number=index + 1,
        source_record_id=f"APP-L000000-0000-{index:03d}",
        source_schema_version="1.0",
        client_ref="L000000",
        payload={
            "event_uid": f"APP-L000000-0000-{index:03d}",
            "login_id": "L000000",
            "device_id": "DEV-1",
            "action": act,
            "app_version": "6.0.1",
            "platform": "ANDROID",
            **({"screen": screen} if screen is not None else {}),
        },
        client_id="C000000",
        timestamp_utc=START + timedelta(minutes=index),
        calendar_timezone="Asia/Almaty",
        event_type=None,
        event_id=None,
    )


def build_one_session(actions) -> MappedRecord:
    sessionizer = Sessionizer(
        load_sessionization_config(
            CONFIG / "sessionization.yaml",
            load_source_contracts(CONFIG / "source_contracts.yaml"),
            load_event_mapping(
                CONFIG / "event_mapping.yaml",
                load_source_contracts(CONFIG / "source_contracts.yaml"),
            ),
        ),
        session_gap=SESSION_GAP,
        max_values_per_field=MAX_VALUES,
        monitor=DataQualityMonitor(),
    )
    built = [r for r in sessionizer.sessionize(actions) if r.event_type == SESSION_EVENT_TYPE]
    assert len(built) == 1
    return built[0], sessionizer


# Порядок обхода намеренно не совпадает с алфавитным: иначе тест не отличил
# бы хронологию от сортировки, и подмена одного другим прошла бы незамеченной.
VISITED = (
    "WALLET", "HOME", "TRANSFERS", "ATM_MAP", "BONUS", "CARDS",
    "SETTINGS", "DEPOSITS", "ANALYTICS", "LOANS", "PROFILE", "SUPPORT",
)


def apply_field_policies(session: MappedRecord) -> ProjectedRecord:
    """Прогнать поля сессии через §21 — тот компонент, что владеет обрезкой."""
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    mapping = load_event_mapping(CONFIG / "event_mapping.yaml", registry)
    schema, policy = load_feature_schema(
        CONFIG / "feature_schema.yaml", registry, mapping.event_types
    )
    projected = list(
        FeatureProjector(schema, policy, registry, monitor=DataQualityMonitor()).project([session])
    )
    return list(
        FieldPolicies(
            schema, default_max_values=MAX_VALUES, monitor=DataQualityMonitor()
        ).apply(projected)
    )[0]


def test_sessionizer_emits_the_full_screen_list():
    """Sessionizer список не режет — обрезкой владеет §21 (2.14).

    Одно правило в двух местах уже дважды оборачивалось расхождением, поэтому
    здесь Sessionizer только считает и отдаёт, а лимит применяет один
    компонент.
    """
    actions = [action(0, None, act="LOGIN")]
    actions += [action(index + 1, screen) for index, screen in enumerate(VISITED)]

    session, sessionizer = build_one_session(actions)

    assert list(session.payload["screens"]) == list(VISITED)
    assert session.payload["screens_count"] == len(VISITED)
    # Лимит Sessionizer знает — но только чтобы поднять аномалию §33.13.
    assert sessionizer.report.summary()["sessions_over_screen_limit"] == 1


def test_screens_count_survives_truncation_by_field_policies():
    """`screens_count` — величина поведения, `screens` — усечённый список.

    Инвариант межкомпонентный: Sessionizer считает счётчик по полному
    набору, §21 режет список, и счётчик обязан это пережить. §19 бакетизирует
    именно `screens_count`, и приравняв его к длине усечённого списка, мы
    объявили бы, что клиент, обошедший двенадцать экранов, обошёл восемь.
    """
    actions = [action(0, None, act="LOGIN")]
    actions += [action(index + 1, screen) for index, screen in enumerate(VISITED)]

    session, _ = build_one_session(actions)
    limited = apply_field_policies(session)

    # После проекции счётчик носит своё конечное имя: §19 бакетизирует
    # `screens_count_bucket`, и значение под ним обязано быть полным.
    assert len(limited.fields["screens"]) == MAX_VALUES
    assert limited.fields["screens_count_bucket"] == len(VISITED)


def test_truncation_keeps_the_earliest_screens_in_chronological_order():
    """Обрезка оставляет первые **по времени** экраны, а не первые по алфавиту.

    Порядок для экранов объявлен значимым (§21 п.3), поэтому усечение — это
    «начало пути», а не выборка. Сортировка тоже дала бы детерминированный
    результат и детерминированно неверный.
    """
    actions = [action(0, None, act="LOGIN")]
    actions += [action(index + 1, screen) for index, screen in enumerate(VISITED)]

    session, _ = build_one_session(actions)
    screens = list(apply_field_policies(session).fields["screens"])

    assert screens == list(VISITED[:MAX_VALUES])
    assert screens != sorted(screens), "порядок совпал с алфавитным — тест ничего не проверяет"
    assert session.payload["first_screen"] == VISITED[0]
    # Последний экран — последний посещённый, а не последний уцелевший после
    # обрезки: иначе поле означало бы «конец усечённого списка».
    assert session.payload["last_screen"] == VISITED[-1]


# --------------------------------------------------------------------------- #
# Границы сессии — §20.1
# --------------------------------------------------------------------------- #


def build_sessions(actions, sessionizer=None):
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    mapping = load_event_mapping(CONFIG / "event_mapping.yaml", registry)
    sessionizer = sessionizer or Sessionizer(
        load_sessionization_config(CONFIG / "sessionization.yaml", registry, mapping),
        session_gap=SESSION_GAP,
        max_values_per_field=MAX_VALUES,
        monitor=DataQualityMonitor(),
    )
    return [r for r in sessionizer.sessionize(actions) if r.event_type == SESSION_EVENT_TYPE]


def at(minutes: int, screen: str | None, *, act: str = "screen_open") -> MappedRecord:
    record = action(minutes, screen, act=act)
    return type(record)(**{**record.__dict__, "timestamp_utc": START + timedelta(minutes=minutes)})


def test_login_starts_a_new_session_even_without_a_gap():
    """Явная граница режет сессию, даже если разрыв меньше `session_gap` (§20.1).

    Вход в приложение — новая сессия по смыслу, а не продолжение прежней.
    Без этого правила два входа подряд слились бы в одну сессию двойной
    длительности.
    """
    actions = [
        at(0, None, act="LOGIN"),
        at(1, "HOME"),
        at(2, None, act="LOGIN"),
        at(3, "CARDS"),
    ]

    sessions = build_sessions(actions)

    assert len(sessions) == 2
    assert [s.payload["action_count"] for s in sessions] == [2, 2]


def test_gap_longer_than_session_gap_splits_the_session():
    """Разрыв больше `session_gap` начинает новую сессию (§20.1)."""
    actions = [
        at(0, None, act="LOGIN"),
        at(5, "HOME"),
        at(5 + 31, "CARDS"),
    ]

    sessions = build_sessions(actions)

    assert len(sessions) == 2
    assert [s.payload["duration_seconds"] for s in sessions] == [5 * 60, 0]


def test_gap_within_session_gap_keeps_one_session():
    """Граница по разрыву не срабатывает раньше времени: 29 минут — та же сессия."""
    actions = [
        at(0, None, act="LOGIN"),
        at(5, "HOME"),
        at(5 + 29, "CARDS"),
    ]

    sessions = build_sessions(actions)

    assert len(sessions) == 1
    assert sessions[0].payload["duration_seconds"] == 34 * 60


def test_repeated_screens_count_once():
    """Счётчик уникальных, а не действий: возврат на тот же экран — не новый экран."""
    actions = [action(0, None, act="LOGIN")]
    actions += [action(index, "HOME") for index in range(1, 6)]
    actions.append(action(6, "CARDS"))

    session, sessionizer = build_one_session(actions)

    assert session.payload["screens_count"] == 2
    assert session.payload["action_count"] == 7
    assert sessionizer.report.summary()["sessions_over_screen_limit"] == 0
