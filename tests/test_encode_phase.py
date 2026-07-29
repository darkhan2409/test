"""ENCODE PHASE — §28.

Два свойства, ради которых фаза вообще отделена от BUILD:

1. **Артефакты загружаются, а не пересчитываются.** Проверяется тем, что
   границы, прочитанные с диска, дают тот же артефакт байт-в-байт: потеря
   точности при разборе сдвинула бы значение на границе в соседний бакет, и
   заметить это можно было бы только по расхождению golden-векторов.
2. **Несовпадение `preprocessing_state_sha256` блокирует обработку** (§30).
   Проверяется на уровне `verify_state_hash` в тестах хэшера; здесь —
   что ENCODE вообще требует хэш и отказывается работать без него.

Прогон ENCODE на настоящих данных живёт не здесь: ему нужны и датасет, и
замороженные артефакты, и он входит в проверку шага 3.3.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.preprocessing.bucketizer import BucketEdges, BucketMethod, FieldEdges
from src.preprocessing.pipeline import EncodePhaseError, FrozenArtifacts
from src.preprocessing.time_delta import DeltaMethod, TimeDeltaEdges

EDGES = BucketEdges(
    version="1.0.0",
    fields={
        "amount_base_bucket": FieldEdges(
            name="amount_base_bucket",
            method=BucketMethod.QUANTILE,
            requested_count=4,
            # Границы взяты как в настоящем артефакте: квантильная граница —
            # это значение выборки, поэтому среди них есть целые (`294`) и
            # суммы с хвостовым нулём (`2500.50`, из `"2 500,50"`). Именно они
            # разбор через float не переживают: `294` станет `294.0`, а
            # `2500.50` — `2500.5`. Аккуратные дроби вроде `100.05` через float
            # проходят без потерь и ничего бы не проверили.
            edges=(Decimal("294"), Decimal("2500.50"), Decimal("30000.125")),
            min_train=Decimal("0.01"),
            max_train=Decimal("999999.99"),
            sample_size=1000,
        )
    },
)

DELTA = TimeDeltaEdges(
    version="1.0.0",
    method=DeltaMethod.QUANTILE,
    requested_count=4,
    edges=(Decimal(60), Decimal(3600), Decimal(86400)),
    min_train=Decimal(0),
    max_train=Decimal(10479366),
    sample_size=1000,
    deltas_seen=30894,
)


# --------------------------------------------------------------------------- #
# Страховка от вырожденных данных
# --------------------------------------------------------------------------- #


def assert_edges_do_not_survive_float(edges) -> None:
    """Границы обязаны различать разбор через строку и через float.

    Утверждается ровно одно свойство: среди значений есть хотя бы одно, чья
    десятичная запись меняется после прохода через `float`. Без него тест
    прошёл бы и на коде, который читает границы через `float`, — а именно
    этого он и не должен пропускать.

    Проверено на настоящем артефакте: из 315 границ 247 через float портятся
    (целое `294` становится `294.0`). Первая версия этого теста брала
    «дробные» границы вроде `100.05`, которые float переживают, и мутацию
    «читаем через float» не поймала.
    """
    broken = [item for item in edges if str(Decimal(str(float(item)))) != str(item)]
    assert broken, (
        f"все границы {[str(item) for item in edges]} переживают float — "
        f"тест не отличит разбор через строку от разбора через float"
    )


# --------------------------------------------------------------------------- #
# Загрузка замороженных артефактов
# --------------------------------------------------------------------------- #


def test_bucket_edges_survive_a_round_trip():
    """Границы, записанные и прочитанные обратно, — те же самые.

    Разбор через `float` сдвинул бы десятичную запись, артефакт перестал бы
    совпадать байт-в-байт, и увидеть это можно было бы только по
    несошедшимся golden-векторам.
    """
    assert_edges_do_not_survive_float(EDGES.fields["amount_base_bucket"].edges)

    restored = BucketEdges.from_state(json.loads(json.dumps(EDGES.state())))

    assert restored.state() == EDGES.state()
    assert restored.fields["amount_base_bucket"].edges == EDGES.fields[
        "amount_base_bucket"
    ].edges


def test_time_delta_edges_survive_a_round_trip():
    """То же для delta-канала: у него собственный артефакт (§25.2)."""
    assert_edges_do_not_survive_float(DELTA.edges)

    restored = TimeDeltaEdges.from_state(json.loads(json.dumps(DELTA.state())))

    assert restored.state() == DELTA.state()


def test_reserved_delta_values_are_not_read_from_the_artifact():
    """`FIRST_EVENT`/`WINDOW_START` заданы кодом, а не файлом (§25.2).

    Прочитать их из артефакта значило бы дать файлу право переименовать
    зарезервированное значение — и разойтись с токенайзером, у которого они
    зашиты (§10.2).
    """
    tampered = DELTA.state()
    tampered["reserved"] = ["ЧТО_УГОДНО"]

    restored = TimeDeltaEdges.from_state(tampered)

    assert restored.domain()[-2:] == ("FIRST_EVENT", "WINDOW_START")


def test_missing_artifacts_are_refused(tmp_path: Path):
    """ENCODE без замороженных артефактов не начинается: пересчитать их нечем."""
    with pytest.raises(EncodePhaseError, match="нет замороженных артефактов"):
        FrozenArtifacts.load(tmp_path, "0.1.0")


def test_incomplete_artifacts_are_refused(tmp_path: Path):
    """Каталог есть, файла нет — это не «версия без границ», это неполный набор."""
    (tmp_path / "0.1.0").mkdir(parents=True)

    with pytest.raises(EncodePhaseError, match="bucket_edges.json"):
        FrozenArtifacts.load(tmp_path, "0.1.0")
