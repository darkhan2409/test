"""Источник `app_logs` — логи мобильного приложения.

Это самый «сырой» источник: одна строка на каждое касание экрана. Ни одна из
них не является бизнес-событием сама по себе — их обязан склеить Sessionizer
(§20) в `APP_SESSION` по разрыву `session_gap`.

Время — наивное локальное, без зоны и без смещения: приложение пишет то, что
показывают часы устройства (§12, п.6).

Push-уведомления идут отдельными строками вне сессий: это самостоятельное
событие, а не действие пользователя.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

from ..catalogs import APP_PLATFORMS, APP_VERSIONS, PUSH_TEMPLATES, SCREENS
from ..clients import SyntheticClient
from ..config import GeneratorConfig
from ..edge_cases import Case, CaseLog, CasePlan, EventStream, has, planned_count
from ..records import RawRecord
from ..rng import derive_rng, weighted_choice
from ..timing import draw_local_datetime, month_start, to_local, to_utc

SOURCE = "app_logs"
TIME_PATTERN = "%Y-%m-%dT%H:%M:%S"

_SESSIONS_PER_MONTH = 0.35
_PUSH_PER_MONTH = 0.8

# Пауза между действиями внутри сессии — заведомо меньше session_gap (30 мин),
# иначе Sessionizer справедливо разрежет её на две сессии.
_MIN_GAP_SECONDS = 3
_MAX_GAP_SECONDS = 240

# Сессия, пересекающая T: начинается до отсечки, продолжается после.
_CROSSING_START_BEFORE_T = timedelta(minutes=25)
_CROSSING_STEP = timedelta(minutes=5)
_CROSSING_ACTIONS = 12


def generate(
    clients: list[SyntheticClient],
    config: GeneratorConfig,
    plan: CasePlan,
    case_log: CaseLog,
) -> Iterator[RawRecord]:
    case_log.record(
        Case.NAIVE_TIMESTAMP,
        source=SOURCE,
        client_id="-",
        record_id="-",
        note="весь источник пишет локальное время без зоны и смещения",
    )
    for client in clients:
        yield from _sessions(client, config)
        yield from _pushes(client, config)
        yield from _crossing_session(client, plan, case_log)


def _sessions(client: SyntheticClient, config: GeneratorConfig) -> Iterator[RawRecord]:
    rng = derive_rng(config.seed, SOURCE, "session", client.client_id)
    active_from = client.active_from(config)
    if active_from > config.history_end:
        return

    platform = weighted_choice(rng, APP_PLATFORMS)
    months = max((config.history_end - active_from).days / 30.44, 0.5)
    natural = max(1, round(months * _SESSIONS_PER_MONTH * client.activity * config.volume_scale))
    session_count = planned_count(client, EventStream.APP_SESSIONS, natural)

    for session_index in range(session_count):
        started = draw_local_datetime(rng, client.timezone, active_from, config.history_end)
        app_version = rng.choice(APP_VERSIONS)
        moment = started
        step = 0

        for action, screen in _session_script(rng):
            yield _action_record(
                client,
                record_id=f"APP-{client.login_id}-{session_index:04d}-{step:03d}",
                moment=moment,
                action=action,
                screen=screen,
                app_version=app_version,
                platform=platform,
            )
            step += 1
            moment += timedelta(seconds=rng.randint(_MIN_GAP_SECONDS, _MAX_GAP_SECONDS))


def _session_script(rng) -> list[tuple[str, str | None]]:
    """Последовательность действий одной сессии: вход, хождение по экранам,
    иногда отправка формы."""
    script: list[tuple[str, str | None]] = [("LOGIN", None)]
    for _ in range(rng.randint(3, 14)):
        screen = rng.choice(SCREENS)
        script.append(("screen_open", screen))
        if rng.random() < 0.55:
            script.append(("button_click", screen))
        if rng.random() < 0.18:
            script.append(("form_submit", screen))
    return script


def _pushes(client: SyntheticClient, config: GeneratorConfig) -> Iterator[RawRecord]:
    rng = derive_rng(config.seed, SOURCE, "push", client.client_id)
    active_from = client.active_from(config)
    if active_from > config.history_end:
        return

    months = max((config.history_end - active_from).days / 30.44, 0.5)
    natural = max(1, round(months * _PUSH_PER_MONTH * client.activity * config.volume_scale))
    count = planned_count(client, EventStream.APP_PUSH, natural)

    for index in range(count):
        moment = draw_local_datetime(rng, client.timezone, active_from, config.history_end)
        payload = {
            "event_uid": f"APP-{client.login_id}-PUSH-{index:04d}",
            "login_id": client.login_id,
            "device_id": client.device_id,
            "ts_local": moment.strftime(TIME_PATTERN),
            "action": "push_received",
            "template_id": rng.choice(PUSH_TEMPLATES),
            "app_version": rng.choice(APP_VERSIONS),
            "platform": weighted_choice(rng, APP_PLATFORMS),
        }
        yield RawRecord(
            source=SOURCE,
            partition_date=month_start(to_utc(moment).date()),
            sort_key=str(payload["event_uid"]),
            payload=payload,
        )


def _crossing_session(
    client: SyntheticClient, plan: CasePlan, case_log: CaseLog
) -> Iterator[RawRecord]:
    """Сессия, начатая до T и продолженная после (§14.1).

    Препроцессинг обязан отрезать действия после T и пересчитать summary
    только по доступной части — длительность такой сессии нельзя брать целиком.
    """
    if not has(client, Case.SESSION_CROSSING_T):
        return

    started = to_local(plan.boundary_at_t - _CROSSING_START_BEFORE_T, client.timezone)
    for step in range(_CROSSING_ACTIONS):
        moment = started + _CROSSING_STEP * step
        record_id = f"APPX-{client.login_id}-cross-{step:03d}"
        if step == 0:
            case_log.record(
                Case.SESSION_CROSSING_T,
                source=SOURCE,
                client_id=client.client_id,
                record_id=record_id,
                note=(
                    f"сессия из {_CROSSING_ACTIONS} действий начинается за "
                    f"{int(_CROSSING_START_BEFORE_T.total_seconds() // 60)} мин до T "
                    "и продолжается после"
                ),
            )
        yield _action_record(
            client,
            record_id=record_id,
            moment=moment,
            action="LOGIN" if step == 0 else "screen_open",
            screen=None if step == 0 else SCREENS[step % len(SCREENS)],
            app_version=APP_VERSIONS[-1],
            platform="ANDROID",
        )


def _action_record(
    client: SyntheticClient,
    *,
    record_id: str,
    moment: datetime,
    action: str,
    screen: str | None,
    app_version: str,
    platform: str,
) -> RawRecord:
    payload: dict[str, object] = {
        "event_uid": record_id,
        "login_id": client.login_id,
        "device_id": client.device_id,
        "ts_local": moment.strftime(TIME_PATTERN),
        "action": action,
        "app_version": app_version,
        "platform": platform,
    }
    if screen is not None:
        payload["screen"] = screen

    return RawRecord(
        source=SOURCE,
        partition_date=month_start(to_utc(moment).date()),
        sort_key=record_id,
        payload=payload,
    )
