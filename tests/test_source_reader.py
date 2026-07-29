"""SourceReader — §4 и §29 п.1.

Две критичные проверки, обе из тех, что нельзя увидеть глазами.

**Порядок партиций.** §29 п.1 требует обхода файлов по каноническому пути.
Ошибка здесь тихая: на машине разработчика `glob` вернёт файлы в удобном
порядке, а на другой файловой системе — в другом, и разойдётся только
итоговый `preprocessing_state_sha256`. Проверка 4.1 (single vs multi-worker)
стоит на этом свойстве напрямую и без теста показала бы «не совпало» без
указания места.

**Маршрутизация нарушений.** §34 запрещает терять записи молча: у каждой
отброшенной должен быть reason code и метрика. Ошибка в маршрутизации не
роняет прогон — она просто уменьшает число записей на выходе, и заметить
это можно только по отчёту, который никто не сверяет построчно.

Контракты берутся настоящие (`config/source_contracts.yaml`), а данные —
временные: так тест заодно сторожит сам конфиг.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.preprocessing.core.monitor import DataQualityMonitor, Metric, Total
from src.preprocessing.core.quarantine import Quarantine, ReasonCode
from src.preprocessing.schema import load_source_contracts
from src.preprocessing.source_reader import SourceContractError, SourceReader

UTC = timezone.utc
RUN_AT = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)
CONTRACTS = Path("config/source_contracts.yaml")

VALID_RATE = {
    "rate_date": "2026-01-05",
    "currency": "USD",
    "rate_to_kzt": "480.1234",
    "source": "NBK",
}


@pytest.fixture(name="registry")
def registry_fixture():
    return load_source_contracts(CONTRACTS)


def make_reader(registry) -> tuple[SourceReader, DataQualityMonitor, Quarantine]:
    monitor = DataQualityMonitor()
    quarantine = Quarantine(monitor, processing_time=RUN_AT, pipeline_version="0.1.0")
    return SourceReader(registry, monitor=monitor, quarantine=quarantine), monitor, quarantine


def build_raw(root: Path, registry, partitions: dict[str, list[str]]) -> Path:
    """Разложить партиции по каталогам источников.

    Каталоги создаются для всех источников контракта: отсутствие каталога —
    отдельная блокирующая ошибка, и она не должна мешать этим тестам.
    """
    for source in registry.sources:
        (root / source).mkdir(parents=True, exist_ok=True)
    for relative, lines in partitions.items():
        path = root / relative
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8") if lines else b"")
    return root


# --------------------------------------------------------------------------- #
# Порядок обхода — §29 пп. 1, 2
# --------------------------------------------------------------------------- #


def test_partition_order_ignores_filesystem_order(tmp_path, registry):
    """Порядок партиций задаётся каноническим путём, а не порядком создания.

    Файлы создаются намеренно в обратном порядке: если бы обход полагался на
    то, что вернула файловая система, здесь он и развалился бы.
    """
    names = ["2026-03-01.jsonl", "2026-01-01.jsonl", "2026-02-01.jsonl"]
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {f"fx_rates/{name}": [json.dumps(VALID_RATE)] for name in names}
        | {f"core_payments/{name}": [] for name in reversed(names)},
    )

    reader, _, _ = make_reader(registry)
    discovered = [partition.canonical_path for partition in reader.discover_partitions(raw)]

    assert discovered == sorted(discovered)
    assert discovered[:3] == [
        "core_payments/2026-01-01.jsonl",
        "core_payments/2026-02-01.jsonl",
        "core_payments/2026-03-01.jsonl",
    ]
    # Путь относительный и в POSIX-виде: абсолютный содержал бы каталог машины,
    # а `\` из Windows сортировался бы иначе, чем `/`.
    assert all("\\" not in path and not Path(path).is_absolute() for path in discovered)


def test_partition_order_survives_an_unsorted_filesystem(tmp_path, registry, monkeypatch):
    """Порядок задаёт сортировка в коде, а не порядок выдачи файловой системы.

    Без подмены `glob` этот тест бесполезен: NTFS отдаёт записи каталога уже
    отсортированными, и убрать сортировку из кода можно, ничего здесь не сломав.
    Ровно так ошибка и доживает до другой машины — проверено мутацией, которая
    без этой подмены выживала.
    """
    names = ["2026-01-01.jsonl", "2026-02-01.jsonl", "2026-03-01.jsonl"]
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {f"fx_rates/{name}": [json.dumps(VALID_RATE)] for name in names},
    )

    original_glob = Path.glob
    monkeypatch.setattr(
        Path, "glob", lambda self, pattern: reversed(sorted(original_glob(self, pattern)))
    )

    reader, _, _ = make_reader(registry)
    discovered = [item.canonical_path for item in reader.discover_partitions(raw)]

    assert discovered == [f"fx_rates/{name}" for name in names]


def test_partition_order_is_stable_across_calls(tmp_path, registry):
    """Повторный обход даёт тот же список — иначе двум воркерам достанется
    разный план работ."""
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {f"fx_rates/2026-0{index}-01.jsonl": [json.dumps(VALID_RATE)] for index in range(1, 6)},
    )

    reader, _, _ = make_reader(registry)
    first = [item.canonical_path for item in reader.discover_partitions(raw)]
    second = [item.canonical_path for item in reader.discover_partitions(raw)]

    assert first == second


def test_records_follow_partition_and_line_order(tmp_path, registry):
    """Чтение идёт по партициям в каноническом порядке, внутри — по строкам.

    Это тот самый порядок, который §29 п.3 требует сохранить при слиянии
    воркеров, поэтому он проверяется вместе с порядком файлов.
    """
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {
            "fx_rates/2026-02-01.jsonl": [
                json.dumps({**VALID_RATE, "rate_date": "2026-02-01", "currency": code})
                for code in ("USD", "EUR")
            ],
            "fx_rates/2026-01-01.jsonl": [
                json.dumps({**VALID_RATE, "rate_date": "2026-01-01", "currency": code})
                for code in ("USD", "EUR")
            ],
        },
    )

    reader, _, _ = make_reader(registry)
    order = [record.source_record_id for record in reader.read_all(raw)]

    assert order == [
        "2026-01-01|USD",
        "2026-01-01|EUR",
        "2026-02-01|USD",
        "2026-02-01|EUR",
    ]


def test_unknown_source_directory_is_blocking(tmp_path, registry):
    """Каталог без Source Contract останавливает прогон.

    Молча прочитать данные, о которых не договаривались, — та же ошибка, что
    молча принять новое поле (§4).
    """
    raw = build_raw(tmp_path / "raw", registry, {})
    (raw / "shadow_source").mkdir()

    with pytest.raises(SourceContractError, match="без Source Contract"):
        make_reader(registry)[0].discover_partitions(raw)


# --------------------------------------------------------------------------- #
# Маршрутизация нарушений — §4, §34, §33
# --------------------------------------------------------------------------- #


def test_contract_violations_are_quarantined_with_reason_and_metric(tmp_path, registry):
    """Каждая отброшенная запись получает reason code и метрику §33.2.

    Проверяются все виды нарушений сразу: пропуск любого из них означает
    молча потерянную запись, а это ровно то, что §34 запрещает.
    """
    broken = [
        json.dumps(VALID_RATE),                                        # 1 валидная
        "{это не json",                                                # 2
        json.dumps([VALID_RATE]),                                      # 3 не объект
        json.dumps({**VALID_RATE, "currency": "EUR", "surprise": 1}),  # 4 поле вне контракта
        json.dumps({k: v for k, v in VALID_RATE.items() if k != "source"}),  # 5 нет поля
        json.dumps({**VALID_RATE, "currency": "CHF", "rate_to_kzt": 480.12}),  # 6 не тот тип
        json.dumps({**VALID_RATE, "currency": ""}),                    # 7 пустой ключ
        json.dumps({**VALID_RATE, "currency": "A|B"}),                 # 8 разделитель в ключе
        json.dumps({**VALID_RATE, "currency": "JPY", "source": None}),  # 9 null при nullable: false
    ]
    raw = build_raw(tmp_path / "raw", registry, {"fx_rates/2026-01-01.jsonl": broken})

    reader, monitor, quarantine = make_reader(registry)
    kept = list(reader.read_all(raw))
    report = monitor.report()

    assert len(kept) == 1
    assert report["totals"][str(Total.RECORDS_READ)] == len(broken)
    assert quarantine.counts_by_reason() == {str(ReasonCode.SOURCE_CONTRACT_VIOLATION): 8}
    # Знаменатель метрики (§33.2) заполняется позже по цепочке, поэтому
    # сверяется числитель: он обязан совпасть с числом отбраковок.
    assert report["metrics"][str(Metric.SCHEMA_VIOLATION_RATE)]["count"] == 8
    assert report["totals"][str(Total.QUARANTINED)] == 8


def test_boolean_is_not_accepted_as_integer(tmp_path, registry):
    """`true` не проходит проверку типа `integer`.

    В Python `bool` — подкласс `int`, поэтому наивная проверка `isinstance`
    пропустила бы `"payload_version": true`. Дальше по цепочке это стало бы
    версией `1` при сравнении конфликтующих дублей (§9.3).
    """
    record = {
        "source_record_id": "CP-1",
        "client_ref": "000000",
        "branch_region": "ALMATY",
        "operation_time": "15.01.2026 14:30",
        "op_code": "PMT",
        "amount": "1000,00",
        "currency": "KZT",
        "counterparty_ref": "KZ000000000001",
        "payload_version": True,
        "loaded_at": "2026-01-15T16:30:00Z",
    }
    raw = build_raw(
        tmp_path / "raw", registry, {"core_payments/2026-01-01.jsonl": [json.dumps(record)]}
    )

    reader, _, quarantine = make_reader(registry)

    assert list(reader.read_all(raw)) == []
    assert quarantine.counts_by_reason() == {str(ReasonCode.SOURCE_CONTRACT_VIOLATION): 1}


def test_schema_version_mismatch_has_its_own_reason(tmp_path, registry):
    """Источник, уехавший на другую версию схемы, — отдельная причина (§34).

    Смешать её с нарушением контракта значило бы чинить не то: контракт цел,
    несовместима версия.
    """
    record = {
        "rec_id": "CRD-1",
        "cardholder_id": "CH000000",
        "txn_time_ms": 1_700_000_000_000,
        "op": "PUR",
        "schema_version": "2.0",
        "terminal_country": "KZ",
    }
    raw = build_raw(
        tmp_path / "raw", registry, {"card_processing/2026-01-01.jsonl": [json.dumps(record)]}
    )

    reader, _, quarantine = make_reader(registry)

    assert list(reader.read_all(raw)) == []
    assert quarantine.counts_by_reason() == {str(ReasonCode.INCOMPATIBLE_SCHEMA_VERSION): 1}


def test_unknown_field_raises_alert_and_names_the_field(tmp_path, registry):
    """§4 требует alert, а не только метрику.

    `schema_violation_rate` скажет «стало хуже», но не назовёт поле, из-за
    которого источник больше не соответствует контракту.
    """
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {"fx_rates/2026-01-01.jsonl": [json.dumps({**VALID_RATE, "surprise": 1})]},
    )

    reader, _, _ = make_reader(registry)
    list(reader.read_all(raw))

    assert reader.schema_alerts() == {"fx_rates": ["surprise"]}


def test_missing_and_invalid_values_survive_the_reader(tmp_path, registry):
    """Ридер проверяет схему, а не значения.

    `null` — это MISSING (§15.2), `"abc"` — задача NumericValidator (§17),
    отсутствие необязательного ключа — «поле неприменимо» (§15.1). Отбраковать
    их здесь значило бы лишить §15 и §17 их же краевых случаев.
    """
    records = [
        {
            "source_record_id": "CP-1",
            "client_ref": "000000",
            "branch_region": "ALMATY",
            "operation_time": "15.01.2026 14:30",
            "op_code": "PMT",
            "amount": None,
            "currency": "",
            "counterparty_ref": "KZ000000000001",
            "payload_version": 1,
            "loaded_at": "2026-01-15T16:30:00Z",
        },
        {
            "source_record_id": "CP-2",
            "client_ref": "000000",
            "branch_region": "ALMATY",
            "operation_time": "15.01.2026 14:31",
            "op_code": "PMT",
            "amount": "abc",
            "currency": "тенге",
            "direction": "N/A",
            "counterparty_ref": "KZ000000000002",
            "payload_version": 1,
            "loaded_at": "2026-01-15T16:30:00Z",
        },
    ]
    raw = build_raw(
        tmp_path / "raw",
        registry,
        {"core_payments/2026-01-01.jsonl": [json.dumps(item) for item in records]},
    )

    reader, _, quarantine = make_reader(registry)
    kept = list(reader.read_all(raw))

    assert [record.source_record_id for record in kept] == ["CP-1", "CP-2"]
    assert kept[0].payload["amount"] is None
    assert "direction" not in kept[0].payload
    assert quarantine.summary()["total"] == 0


# --------------------------------------------------------------------------- #
# PII в идентификаторе записи — §4, §23, §13
# --------------------------------------------------------------------------- #


def test_primary_key_cannot_be_a_direct_identifier(tmp_path: Path):
    """Прямой идентификатор не может быть первичным ключом источника.

    `source_record_id` склеивается из `primary_key` и уезжает к токенайзеру
    внутри `ordering_key` (§13, §5 п.7) — значением поля он не становится и
    токеном не будет, но в выходном файле лежит открытым текстом. §7 п.4
    уже запрещает PII в поле клиента; здесь та же дыра со стороны ключа
    записи, и на синтетике она безобидна только потому, что генератор
    нумерует записи сам. Боевой источник волен нумеровать чем угодно.

    Проверка стоит на модели контракта: объявить такой источник нельзя, а не
    «нужно не забыть проверить».
    """
    document = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    contract = document["sources"]["core_payments"]
    key = contract["primary_key"][0]
    assert contract["columns"][key]["pii"] == "none", "ключ уже помечен — тест ничего не проверит"
    contract["columns"][key]["pii"] = "direct_identifier"

    path = tmp_path / "contracts.yaml"
    path.write_bytes(yaml.safe_dump(document, allow_unicode=True).encode("utf-8"))

    with pytest.raises(Exception, match="direct_identifier"):
        load_source_contracts(path)
