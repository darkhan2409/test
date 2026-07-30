"""Демо-экран препроцессинга: одна транзакция шаг за шагом.

Запуск:
    streamlit run ui/app.py

Экран только читает то, что оставил прогон (`data/debug/`, `data/prepared/`), и
показывает разницу между входом и выходом каждого шага. Никакой логики
пайплайна здесь нет — иначе она разошлась бы с кодом при первой правке.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da
import steps as S

st.set_page_config(page_title="Препроцессинг: шаг за шагом", layout="wide")

# Крупный шрифт: смотреть будут через проектор.
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 18px; }
      .stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.55; }
      h1 { font-size: 2.0rem !important; }
      h2 { font-size: 1.5rem !important; }
      h3 { font-size: 1.2rem !important; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .stButton button { font-size: 1.0rem; padding: 0.35rem 0.4rem; white-space: nowrap; }
      .diff-table { width: 100%; border-collapse: collapse; font-size: 1.02rem; }
      .diff-table th { text-align: left; padding: 6px 10px; color: #666; font-weight: 600; }
      .diff-table td { padding: 6px 10px; vertical-align: top; border-top: 1px solid #eee; }
      .diff-table td.path { font-family: ui-monospace, monospace; color: #333; width: 26%; }
      .val { font-family: ui-monospace, monospace; }
      .before { background: #fff1f0; }
      .after  { background: #eaf7ea; }
      .added  { background: #eaf7ea; }
      .tag { font-size: 0.85rem; color: #888; }
      .footer { color: #999; font-size: 0.85rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _events() -> dict[str, list[da.EventRef]]:
    return da.events_by_client(da.prepared_events())


@st.cache_data(show_spinner=False)
def _clean() -> list[str]:
    return da.clean_clients()


@st.cache_data(show_spinner=False)
def _traced() -> set[str]:
    return da.traced_clients()


@st.cache_data(show_spinner="Собираю путь записи по трассировке…")
def _trace(source_record_id: str, event_id: str) -> dict[str, dict[str, Any | None]]:
    return da.trace(source_record_id, event_id)


@st.cache_data(show_spinner=False)
def _prepared(event_id: str) -> dict[str, Any] | None:
    return da.prepared_event(event_id)


missing = da.check_ready()
if missing:
    st.title("Препроцессинг: шаг за шагом")
    st.error("Не хватает данных прогона:\n\n" + "\n".join(f"- `{item}`" for item in missing))
    st.markdown("Собрать их: `python -m tools.run_pipeline --debug`")
    st.stop()

all_events = _events()
traced = _traced()
# Трассировка есть только у клиентов из фильтра §1.6 — остальных в выбор не
# берём, иначе экран показал бы «шаг записи не касался» на всех 17 шагах.
by_client = {client: refs for client, refs in all_events.items() if client in traced}
clean = [item for item in _clean() if item in by_client]
if not by_client:
    st.title("Препроцессинг: шаг за шагом")
    st.error(
        "В трассировке нет ни одного клиента, у которого есть события. "
        "Пересоберите данные: `python -m tools.run_pipeline --debug`."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Шапка: выбор клиента и события
# --------------------------------------------------------------------------- #

clients = sorted(by_client)
default_client = clean[-1] if clean else clients[0]

# После старта выбор уезжает в свёрнутый блок: на проекторе важнее, чтобы
# содержимое шага было наверху, а не шапка.
if st.session_state.get("started"):
    st.markdown("#### Препроцессинг: одна транзакция шаг за шагом")
    picker = st.expander("Выбор клиента и события", expanded=False)
else:
    st.title("Препроцессинг: одна транзакция шаг за шагом")
    picker = st.container()

head = picker.columns([1.1, 2.6, 1.0])

with head[0]:
    client = st.selectbox(
        "Клиент",
        clients,
        index=clients.index(st.session_state.get("client", default_client)),
        format_func=lambda item: f"{item} · чистый" if item in clean else item,
        key="client",
    )

refs = by_client[client]
# По умолчанию — операция в валюте: на ней видно больше шагов (нормализация
# валюты, разбор суммы, пересчёт курса). Это выбор того, что показать первым,
# а не подгонка данных: любое другое событие выбирается в том же списке.
preferred = next(
    (item for item in refs if item.currency and item.currency != "KZT"),
    next((item for item in refs if item.event_type == "PAYMENT"), refs[0]),
)

with head[1]:
    event_labels = [item.label() for item in refs]
    default_index = refs.index(preferred)
    if st.session_state.get("_client_shown") != client:
        st.session_state["_client_shown"] = client
        st.session_state["event_label"] = event_labels[default_index]
    label = st.selectbox(
        f"Событие ({len(refs)} шт.)",
        event_labels,
        index=event_labels.index(st.session_state.get("event_label", event_labels[default_index])),
        key="event_label",
    )
    event = refs[event_labels.index(label)]

with head[2]:
    st.write("")
    if st.button("Начать", type="primary", use_container_width=True):
        st.session_state["step"] = 0
        st.session_state["started"] = True

if not st.session_state.get("started"):
    st.info(
        "Выберите клиента и событие, затем нажмите «Начать». "
        f"По умолчанию открыт клиент **{default_client}** — у него нет ни одного "
        "краевого случая, то есть обычный путь без брака."
    )
    st.markdown(f"<div class='footer'>{S.DISCLAIMER}</div>", unsafe_allow_html=True)
    st.stop()

trace = _trace(event.source_record_id, event.event_id)
step_index = int(st.session_state.get("step", 0))
last_index = S.TOTAL  # 0..TOTAL-1 — шаги, TOTAL — итоговая карточка


# --------------------------------------------------------------------------- #
# Карта шагов
# --------------------------------------------------------------------------- #

st.markdown("###### Карта шагов — можно перейти на любой")


def _step_button(position: int, column: Any) -> None:
    with column:
        if position < S.TOTAL:
            step = S.STEPS[position]
            caption = f"{step.position}. {step.short}"
            help_text = f"{step.title} · {step.group} · {step.spec} · пункт плана {step.plan_item}"
        else:
            caption = "Итог"
            help_text = S.SUMMARY_TITLE
        if st.button(
            caption,
            key=f"jump_{position}",
            type="primary" if position == step_index else "secondary",
            use_container_width=True,
            help=help_text,
        ):
            st.session_state["step"] = position
            st.rerun()


# Две строки по девять: в один ряд восемнадцать подписей не влезают, а номер
# без названия на проекторе ничего не подсказывает.
half = (S.TOTAL + 2) // 2
for row_start in (0, half):
    row = st.columns(half)
    for offset, column in enumerate(row):
        position = row_start + offset
        if position <= S.TOTAL:
            _step_button(position, column)


def _render_value(value: Any) -> str:
    if value is da.ABSENT:
        return "<span class='tag'>—</span>"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    elif value is None:
        text = "null"
    else:
        text = str(value)
    if len(text) > 160:
        text = text[:160] + "…"
    return f"<span class='val'>{_escape(text)}</span>"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _diff_table(diffs: list[da.FieldDiff]) -> str:
    rows = []
    for item in diffs:
        if item.kind == "added":
            mark = "появилось"
        elif item.kind == "removed":
            mark = "исчезло"
        else:
            mark = "изменилось"
        rows.append(
            f"<tr><td class='path'>{_escape(item.path)}<br><span class='tag'>{mark}</span></td>"
            f"<td class='before'>{_render_value(item.before)}</td>"
            f"<td class='after'>{_render_value(item.after)}</td></tr>"
        )
    return (
        "<table class='diff-table'><tr><th>поле</th><th>было</th><th>стало</th></tr>"
        + "".join(rows)
        + "</table>"
    )


# --------------------------------------------------------------------------- #
# Шаг
# --------------------------------------------------------------------------- #

if step_index < S.TOTAL:
    step = S.STEPS[step_index]
    st.markdown(
        f"## {step.group} · Шаг {step.position} из {S.TOTAL} — {step.title} ({step.spec})"
    )
    st.caption(f"пункт плана {step.plan_item} · разбор в docs/explained")

    st.markdown(f"**Что делает.** {step.does}")

    rows = trace.get(step.slug or "", {"in": None, "out": None})
    before, after = rows.get("in"), rows.get("out")

    st.markdown("### Было → стало")

    if step.slug is None:
        st.info(step.note)
    elif before is None and after is None:
        st.warning(
            "Эта запись через шаг не проходила — трассировка её здесь не "
            "содержит. Что именно шаг делает с записями, которых касается, "
            "написано ниже."
        )
    else:
        diffs, same = da.compare(before, after)
        if not diffs:
            st.success("Изменений нет: шаг эту запись не менял.")
        else:
            st.markdown(_diff_table(diffs), unsafe_allow_html=True)

        if same:
            with st.expander(f"Поля без изменений ({len(same)})"):
                st.json(same, expanded=False)

        with st.expander("Строки трассировки как есть"):
            columns = st.columns(2)
            with columns[0]:
                st.caption("вход")
                st.json(before or {"—": "строки нет"}, expanded=False)
            with columns[1]:
                st.caption("выход")
                st.json(after or {"—": "строки нет"}, expanded=False)

    st.markdown("### Что проверено")
    if step.note and step.slug is not None:
        st.caption(step.note)
    st.markdown(step.checks)

else:
    # Итоговая карточка
    st.markdown(f"## {S.SUMMARY_TITLE}")
    reader = trace.get(da.READER_SLUG, {})
    raw = (reader.get("in") or reader.get("out") or {}).get("payload", {})
    final = _prepared(event.event_id)

    columns = st.columns(2)
    with columns[0]:
        st.markdown("#### Сырая запись из банковской системы")
        st.json(raw.get("record", raw), expanded=True)
    with columns[1]:
        st.markdown("#### Готовое событие для токенайзера")
        st.json(final or {}, expanded=True)

    st.success(S.SUMMARY_TEXT)

    report = da.run_report()
    if report:
        totals = report.get("totals", {})
        metrics = st.columns(4)
        metrics[0].metric("Прочитано записей", f"{totals.get('records_read', 0):,}".replace(",", " "))
        metrics[1].metric("Событий на выходе", f"{len(da.prepared_events()):,}".replace(",", " "))
        metrics[2].metric("Клиентов", f"{totals.get('clients_processed', 0):,}".replace(",", " "))
        metrics[3].metric("В карантине", report.get("quarantine", {}).get("total", 0))


# --------------------------------------------------------------------------- #
# Навигация
# --------------------------------------------------------------------------- #

st.divider()
nav = st.columns([1, 1, 6])
with nav[0]:
    if st.button("← Назад", disabled=step_index == 0, use_container_width=True):
        st.session_state["step"] = max(0, step_index - 1)
        st.rerun()
with nav[1]:
    if st.button(
        "Следующий шаг →",
        type="primary",
        disabled=step_index >= last_index,
        use_container_width=True,
    ):
        st.session_state["step"] = min(last_index, step_index + 1)
        st.rerun()

st.markdown(
    f"<div class='footer'>{S.DISCLAIMER} · клиент {event.client_id} · "
    f"запись {event.source_record_id} · событие {event.event_id}</div>",
    unsafe_allow_html=True,
)
