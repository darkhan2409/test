"""FieldPolicies — §21, §22, §23.

Проверяется то, чего в модели быть **не должно**, а такие ошибки тихие:
лишнее поле не роняет прогон, оно просто уходит к токенайзеру и попадает
в словарь.

Технический идентификатор — худший случай. `merchant_id` большой мощности:
попав в словарь, он даст десятки тысяч токенов, каждый встреченный один раз.
Модель не сломается — она станет хуже, и понять почему будет не по чему.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocessing.core.monitor import DataQualityMonitor
from src.preprocessing.event_mapper import load_event_mapping
from src.preprocessing.feature_projection import ProjectedRecord, load_feature_schema
from src.preprocessing.field_policies import (
    FieldPolicies,
    FieldPolicyError,
    check_field_policies,
)
from src.preprocessing.schema import load_source_contracts
from src.preprocessing.schema.source_contract import PiiClass

CONFIG = Path("config")
MAX_VALUES = 8


@pytest.fixture(name="setup")
def setup_fixture():
    registry = load_source_contracts(CONFIG / "source_contracts.yaml")
    mapping = load_event_mapping(CONFIG / "event_mapping.yaml", registry)
    schema, _ = load_feature_schema(CONFIG / "feature_schema.yaml", registry, mapping.event_types)
    return schema, registry


def purchase(fields) -> ProjectedRecord:
    return ProjectedRecord(
        source="card_processing",
        partition="card_processing/2026-01-01.jsonl",
        line_number=1,
        source_record_id="CRD-1",
        source_schema_version="1.2",
        client_ref="CH000000",
        payload={},
        client_id="C000000",
        timestamp_utc=None,
        calendar_timezone="Asia/Almaty",
        event_type="CARD_PURCHASE",
        event_id="a" * 32,
        fields=fields,
        schema_section="CARD_PURCHASE",
    )


def apply(schema, record) -> ProjectedRecord:
    policies = FieldPolicies(
        schema, default_max_values=MAX_VALUES, monitor=DataQualityMonitor()
    )
    return list(policies.apply([record]))[0]


def test_technical_identifier_does_not_reach_the_model(setup):
    """§22: `merchant_id` не доходит до токенайзера.

    Поле объявлено в схеме явно (`model_input: false`), а не пропущено молча —
    решение «не показывать модели» должно быть видно в контракте. Здесь
    проверяется, что объявление действительно исполняется.
    """
    schema, _ = setup
    record = purchase({"merchant_category": "GROCERY", "merchant_id": "M-0042"})

    result = apply(schema, record)

    assert "merchant_id" not in result.fields
    assert result.fields["merchant_category"] == "GROCERY"


def test_excluded_field_removal_is_counted(setup):
    """Убранное поле попадает в отчёт: «поля нет» и «поле убрали» — разное."""
    schema, _ = setup
    policies = FieldPolicies(
        schema, default_max_values=MAX_VALUES, monitor=DataQualityMonitor()
    )

    list(policies.apply([purchase({"merchant_id": "M-1"}), purchase({"merchant_id": "M-2"})]))

    assert policies.report.summary()["excluded_by_field"] == {"merchant_id": 2}


def test_multivalue_with_significant_order_keeps_the_earliest(setup):
    """§21 пп. 3–5: у поля со значимым порядком обрезается хвост."""
    schema, _ = setup
    visited = ["WALLET", "HOME", "TRANSFERS", "ATM_MAP", "BONUS", "CARDS",
               "SETTINGS", "DEPOSITS", "ANALYTICS", "LOANS"]
    record = ProjectedRecord(
        source="app_logs", partition="app_logs/2026-01-01.jsonl", line_number=1,
        source_record_id="APP-1", source_schema_version="1.0", client_ref="L000000",
        payload={}, client_id="C000000", timestamp_utc=None, calendar_timezone="Asia/Almaty",
        event_type="APP_SESSION", event_id="b" * 32,
        fields={"screens": list(visited)}, schema_section="APP_SESSION",
    )

    result = apply(schema, record)

    assert result.fields["screens"] == visited[:MAX_VALUES]
    assert result.fields["screens"] != sorted(result.fields["screens"])


def test_multivalue_without_significant_order_is_sorted(setup):
    """§21 п.2: у поля с незначимым порядком значения сортируются.

    Один и тот же набор продуктов обязан давать одну последовательность
    токенов независимо от того, в каком порядке его прислал источник.
    """
    schema, _ = setup
    record = ProjectedRecord(
        source="profile_snapshots", partition="p", line_number=1,
        source_record_id="CIF-1", source_schema_version="1.0", client_ref="CIF000000",
        payload={}, client_id="C000000", timestamp_utc=None, calendar_timezone="Asia/Almaty",
        event_type=None, event_id=None,
        fields={"products": ["LOAN", "CARD", "DEPOSIT"]}, schema_section="PROFILE",
    )

    result = apply(schema, record)

    assert result.fields["products"] == ["CARD", "DEPOSIT", "LOAN"]


def test_direct_identifier_reaching_the_model_is_blocking(setup):
    """§23: PII до токенайзера не доходит.

    Проверка идёт по классификации Source Contract, а не по имени поля:
    «выглядит как телефон» — не критерий, `pii: direct_identifier` — решение
    владельца источника.
    """
    schema, registry = setup
    contract = registry.sources["profile_snapshots"]
    leaking = registry.model_copy(update={"sources": {
        **registry.sources,
        "profile_snapshots": contract.model_copy(update={"columns": {
            **contract.columns,
            "region": contract.columns["region"].model_copy(
                update={"pii": PiiClass.DIRECT_IDENTIFIER}
            ),
        }}),
    }})

    with pytest.raises(FieldPolicyError, match="прямые идентификаторы"):
        check_field_policies(schema, leaking, default_max_values=MAX_VALUES)


def test_real_config_has_no_pii_leak(setup):
    """Настоящая конфигурация проходит — точка отсчёта для мутаций."""
    schema, registry = setup

    check_field_policies(schema, registry, default_max_values=MAX_VALUES)


def test_high_cardinality_strategies_still_exclude_rare_assignment():
    """Сторож на временную гарантию: `RARE` препроцессингом не назначается.

    Сейчас это обеспечено составом `HighCardinalityPolicy`: в нём только
    `exclude` и `pass_to_tokenizer`, а `keep_frequent` из §22 — та самая
    стратегия, которая означала бы «оставить частые, остальное схлопнуть»,
    то есть назначение `RARE` в препроцессинге. Объявить её нечем.

    Гарантия **временная**: §22 эту стратегию описывает, и однажды её могут
    реализовать. Тест падает в этот момент и напоминает, что §14.1
    токенайзера считает `RARE` своей ответственностью, а хвостовой бакет
    §2.2 обязан пережить `min_count`.
    """
    from src.preprocessing.schema.feature_schema import HighCardinalityPolicy

    assert {str(item) for item in HighCardinalityPolicy} == {"exclude", "pass_to_tokenizer"}, (
        "в HighCardinalityPolicy появилась стратегия: если она отсеивает редкие "
        "значения, препроцессинг начал назначать RARE — §22 это запрещает"
    )
