"""Справочники синтетики.

Значения намеренно «грязные»: одна и та же валюта приходит из разных систем
как `KZT`, `kzt`, `тенге` и `398`. Приводить их к канону — работа
CategoryNormalizer (§16), генератор только создаёт разнобой.
"""

from __future__ import annotations

# Регион → IANA timezo
# ne. Регионы с разными смещениями взяты специально:
# до 2024-03-01 Алматы жил в UTC+6, а западные регионы — в UTC+5, и §12.1
# запрещает применять единый UTC+05 к истории задним числом.
REGIONS: dict[str, str] = {
    "ALMATY": "Asia/Almaty",
    "ASTANA": "Asia/Almaty",
    "SHYMKENT": "Asia/Almaty",
    "KOSTANAY": "Asia/Qostanay",
    "AKTOBE": "Asia/Aqtobe",
    "ATYRAU": "Asia/Atyrau",
    "ORAL": "Asia/Oral",
}

REGION_WEIGHTS: dict[str, float] = {
    "ALMATY": 0.30,
    "ASTANA": 0.22,
    "SHYMKENT": 0.16,
    "KOSTANAY": 0.08,
    "AKTOBE": 0.10,
    "ATYRAU": 0.07,
    "ORAL": 0.07,
}

EMPLOYMENT_WEIGHTS: dict[str, float] = {
    "EMPLOYED": 0.62,
    "SELF_EMPLOYED": 0.14,
    "STUDENT": 0.08,
    "RETIRED": 0.10,
    "UNEMPLOYED": 0.06,
}

# Как валюта выглядит в каждой системе. core_payments — «человеческий» ввод,
# card_processing — ISO-4217 numeric.
CURRENCY_WEIGHTS: dict[str, float] = {"KZT": 0.88, "USD": 0.08, "EUR": 0.04}

CURRENCY_ALIASES_TEXT: dict[str, list[str]] = {
    "KZT": ["KZT", "kzt", "тенге", "ТЕНГЕ", "398"],
    "USD": ["USD", "usd", "доллар", "840"],
    "EUR": ["EUR", "евро", "978"],
}

CURRENCY_ISO_NUMERIC: dict[str, str] = {"KZT": "398", "USD": "840", "EUR": "978"}

# Курс к тенге на начало истории; дальше — случайное блуждание.
FX_BASE_RATE: dict[str, float] = {"USD": 470.0, "EUR": 505.0}

# Коды операций источников. Отображение в canonical event_type — забота
# EventMapper (§10), поэтому здесь только сырые коды.
CORE_PAYMENT_OP_WEIGHTS: dict[str, float] = {
    "TRF": 0.45,  # перевод
    "PMT": 0.40,  # платёж
    "LNP": 0.15,  # погашение кредита
}

CARD_OP_WEIGHTS: dict[str, float] = {
    "PUR": 0.78,  # покупка
    "WDR": 0.20,  # снятие наличных
    "BLK": 0.02,  # блокировка карты
}

DIRECTIONS: dict[str, float] = {"OUT": 0.8, "IN": 0.2}

# Голова распределения + длинный хвост: хвостовые значения сами по себе
# окажутся реже min_count и станут RARE у токенайзера.
MERCHANT_CATEGORY_WEIGHTS: dict[str, float] = {
    "GROCERY": 0.26,
    "FUEL": 0.13,
    "RESTAURANT": 0.12,
    "PHARMACY": 0.09,
    "TRANSPORT": 0.08,
    "CLOTHING": 0.07,
    "ELECTRONICS": 0.05,
    "UTILITIES": 0.05,
    "TELECOM": 0.04,
    "ENTERTAINMENT": 0.03,
    "HOTEL": 0.02,
    "AIRLINE": 0.02,
    "BOOKSTORE": 0.010,
    "FLORIST": 0.008,
    "VETERINARY": 0.006,
    "ART_DEALER": 0.004,
    "AQUARIUM_SUPPLIES": 0.002,
    "FALCONRY_SUPPLIES": 0.001,
}

MCC_BY_CATEGORY: dict[str, str] = {
    "GROCERY": "5411",
    "FUEL": "5541",
    "RESTAURANT": "5812",
    "PHARMACY": "5912",
    "TRANSPORT": "4111",
    "CLOTHING": "5651",
    "ELECTRONICS": "5732",
    "UTILITIES": "4900",
    "TELECOM": "4814",
    "ENTERTAINMENT": "7832",
    "HOTEL": "7011",
    "AIRLINE": "4511",
    "BOOKSTORE": "5942",
    "FLORIST": "5992",
    "VETERINARY": "0742",
    "ART_DEALER": "5971",
    "AQUARIUM_SUPPLIES": "5995",
    "FALCONRY_SUPPLIES": "5999",
}

SCREENS: list[str] = [
    "HOME",
    "CARDS",
    "TRANSFERS",
    "PAYMENTS",
    "DEPOSITS",
    "LOANS",
    "PROFILE",
    "SUPPORT",
    "QR",
]

APP_PLATFORMS: dict[str, float] = {"ANDROID": 0.62, "IOS": 0.36, "HUAWEI": 0.02}

APP_VERSIONS: list[str] = ["5.10.2", "5.11.0", "5.12.0", "5.12.3", "6.0.1"]

PRODUCTS: list[str] = ["CARD", "LOAN", "DEPOSIT", "SAVINGS", "INSURANCE", "BROKERAGE"]

# Заведомо длиннее max_values_per_field (8) — для проверки обрезки многозначных
# полей по §21.
EXTENDED_PRODUCTS: list[str] = sorted(
    PRODUCTS + ["PENSION", "TRAVEL_CARD", "VIRTUAL_CARD", "CHILD_CARD", "METALS"]
)

# Категории, которые НИКОГДА не выпадают случайно: они появляются только через
# инъекцию краевых случаев, поэтому их частота предсказуема.
# RARE_ONLY встречается несколько раз до T (реже min_count → RARE у токенайзера),
# UNSEEN_ONLY — только после T (в TRAIN его нет → [UNK]).
RARE_ONLY_CATEGORY = "TAXIDERMY"
UNSEEN_ONLY_CATEGORY = "CRYPTO_EXCHANGE"

MCC_BY_CATEGORY[RARE_ONLY_CATEGORY] = "5999"
MCC_BY_CATEGORY[UNSEEN_ONLY_CATEGORY] = "6051"

PUSH_TEMPLATES: list[str] = [
    "TPL_PAYMENT_REMINDER",
    "TPL_CASHBACK",
    "TPL_SECURITY_ALERT",
    "TPL_MARKETING_OFFER",
    "TPL_LOAN_DUE",
]
