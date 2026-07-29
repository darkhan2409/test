# Регламент preprocessing банковских данных

*Для банковской событийной модели и токенайзера*

- **Статус:** официальный регламент
- **Связанный документ:** `tokenizer_rules_official`
- **Назначение:** определить единые правила преобразования сырых банковских данных в канонические профиль и события, готовые для токенайзера.

## 1. НАЗНАЧЕНИЕ И ГРАНИЦЫ ОТВЕТСТВЕННОСТИ

Preprocessing преобразует разнородные сырые записи банковских систем в стабильный набор подготовленных данных.

Основная цепочка:

```text
сырые источники
→ проверка и очистка
→ нормализация типов
→ timestamp в UTC + локальная timezone
→ канонический client_id
→ cutoff T
→ дедупликация
→ построение canonical event_type
→ sessionization
→ Feature Schema
→ MISSING
→ нормализация категорий
→ проверка числовых значений
→ FX-нормализация
→ bucketization
→ проверка bucket domain
→ профиль на T и life-long признаки
→ единый timeline
→ временные метаданные (local hour/day)
→ подготовленные данные для токенайзера
```

Preprocessing отвечает за:

- контракты входных источников;
- приведение типов;
- нормализацию timestamp и часовых поясов;
- стабильную идентификацию клиента и записи;
- дедупликацию;
- преобразование строк источников в бизнес-события;
- sessionization app-логов;
- применение cutoff time T;
- защиту от data leakage;
- построение и применение Feature Schema;
- нормализацию пропусков;
- нормализацию категориальных значений;
- проверку числовых значений;
- историческую FX-нормализацию сумм;
- расчёт и применение bucket_edges;
- clipping значений вне TRAIN-диапазона;
- подготовку многозначных полей;
- обработку высококардинальных и текстовых полей;
- построение профиля на T;
- расчёт life-long признаков;
- построение временных метаданных;
- сборку единого timeline;
- детерминированность BUILD PHASE;
- версионирование и контроль целостности артефактов;
- data-quality monitoring.

Preprocessing не отвечает за:

- добавление `[USR]` и `[EVT]`;
- создание строковых key/value-токенов;
- сбор token counts;
- `min_count`, `RARE` и Vocabulary;
- назначение `token_id`;
- `[UNK]`, `[PAD]`, `[MASK]`;
- `attention_mask`;
- masking и labels;
- embedding;
- Transformer;
- downstream target и loss.

Эти операции регулируются документом токенайзера и регламентом обучения модели.

## 2. КОНТРАКТ МЕЖДУ PREPROCESSING И ТОКЕНАЙЗЕРОМ

Preprocessing обязан передать токенайзеру:

1. `prepared_profile` — профиль клиента на cutoff T.
2. `prepared_events` — события клиента с `timestamp <= T`.
3. `feature_schema` — поля, порядок, приоритеты и vocabulary policy.
4. `closed_set_domains` — допустимые значения closed-set полей.
5. `bucket_field_domains` — полный configured domain каждого bucket-поля:
   `bucket_0…bucket_N` и `MISSING`.
6. `bucket_metadata` — интервалы bucket values только для decode/observability.
7. `time_delta_edges` — frozen edges для window-relative delta channel, если используется bucket-вариант.
8. `calendar_time_features` — локальные hour/day и timezone metadata.
9. `currency_normalization_config`.
10. `fx_max_staleness`.
11. `sessionization_config`.
12. `field_priority`.
13. `max_values_per_field`.
14. `text_policy`.
15. `cutoff_policy`.
16. `preprocessing_version`.
17. `preprocessing_state_sha256`.
18. data-quality statistics.
19. `lifetime_first` — признак самого раннего известного события клиента (для различения `FIRST_EVENT`/`WINDOW_START`, tokenizer §10.1).

### 2.1. Роль bucket metadata

Numeric bucketization полностью завершена в preprocessing.

Токенайзер получает готовое значение:

```text
amount_base_bucket = bucket_17 | MISSING
```

`bucket_metadata` передаётся токенайзеру только для:

- decode;
- объяснения диапазона;
- model card;
- observability;
- compatibility validation.

Токенайзер не использует numeric bucket edges для encode и не выполняет повторный binning.

### 2.2. Closed-set контракт

Bucket-поля являются `bucket_closed_set`.

Все configured bucket values должны быть переданы токенайзеру как whitelist независимо от TRAIN-частоты.

Это гарантирует, что хвостовой бакет не будет схлопнут в RARE из-за `min_count`.

На входе токенайзера не должно оставаться:

- необработанных `NULL`, `None` или `NaN`;
- строковых сумм;
- ненормализованных валют;
- неизвестного часового пояса;
- событий после T;
- сырых технических ID;
- необработанного свободного текста;
- дубликатов;
- неустойчивого порядка событий;
- значений, требующих numeric parsing, FX или clipping;
- `event_type`, продублированного внутри `fields`.

## 3. ОСНОВНЫЕ ПОНЯТИЯ

### 3.1. Raw record

Исходная строка или сообщение из банковской системы.

Пример:

```text
source_system = core_payments
client_ref = 000123
operation_time = 15.01.2026 14:30
amount = "15 000,50"
currency = "тенге"
```

Raw record не передаётся токенайзеру напрямую.

### 3.2. Canonical event

Нормализованное бизнес-событие с top-level `event_type`.

Пример:

```json
{
  "client_id": "123",
  "event_id": "evt_...",
  "event_type": "TRANSFER",
  "timestamp_utc": "2026-01-15T09:30:00Z",
  "calendar_timezone": "Asia/Almaty",
  "fields": {
    "amount_base_bucket": "bucket_17",
    "currency": "KZT"
  }
}
```

`event_type` не дублируется в `fields`.

Токенайзер читает top-level `event_type` и эмитит его первым после `[EVT]`.

### 3.3. Profile snapshot

Снимок профиля клиента на конкретную дату.

При построении примера выбирается последний снимок с:

```text
profile_time_utc <= T
```

Будущий снимок использовать запрещено.

### 3.4. Cutoff time T

Момент, относительно которого строится модельный пример.

Все входные данные должны быть известны не позднее T.

### 3.5. BUILD PHASE

Одноразовая сборка preprocessing-артефактов на TRAIN:

- mapping-конфигураций;
- bucket_edges;
- time_bucket_edges;
- статистик;
- versioned state;
- content hash.

### 3.6. ENCODE PHASE

Применение замороженных preprocessing-артефактов к каждому TRAIN, Validation, Test или Production-примеру.

## 4. ВХОДНЫЕ ИСТОЧНИКИ И SOURCE CONTRACT

Для каждого источника должен существовать формальный `Source Contract`.

Минимальные поля контракта:

- имя источника;
- владелец источника;
- формат данных;
- схема колонок;
- типы колонок;
- первичный ключ;
- поле клиента;
- event time;
- processing time;
- source_priority (детерминированный ранг источника для timeline tie-break);
- timezone;
- правила обновления;
- правила удаления;
- признак correction/reversal;
- допустимая задержка;
- политика late-arriving records;
- data-retention policy;
- PII classification.

Примеры источников:

- core transactions;
- card processing;
- payments;
- communications;
- mobile app logs;
- internet banking logs;
- CRM;
- profile snapshots;
- product registry;
- loan systems;
- historical FX rates.

Если источник изменил схему без новой версии Source Contract, обработка должна:

- остановить несовместимый pipeline;
- либо отправить записи в quarantine;
- создать data-quality alert.

Молчаливое принятие новых полей и типов запрещено.

## 5. КАНОНИЧЕСКАЯ СХЕМА СОБЫТИЯ

Минимальная структура:

```json
{
  "client_id": "string",
  "event_id": "string",
  "event_type": "string",
  "timestamp_utc": "ISO-8601 UTC",
  "calendar_timezone": "IANA timezone",
  "ordering_key": "string",
  "fields": {},
  "calendar_time_features": {
    "hour_of_day_local": 0,
    "day_of_week_local": 0
  },
  "source_meta": {
    "source_system": "string",
    "source_record_id": "string",
    "source_schema_version": "string"
  },
  "quality_flags": []
}
```

Правила:

1. `client_id` — канонический ID клиента.
2. `event_id` — стабильный ID подготовленного события.
3. `event_type` — top-level утверждённый тип события.
4. `event_type` не дублируется в `fields`.
5. `timestamp_utc` хранит абсолютный instant.
6. `calendar_timezone` задаёт IANA timezone для поведенческих календарных признаков.
7. `ordering_key` — детерминированный ключ сортировки.
8. `fields` — только признаки Feature Schema, без event_type.
9. `calendar_time_features` содержит локальные hour/day.
10. Финальная `delta_from_previous_event` не хранится как готовый model input: она пересчитывается после truncation.
11. `source_meta` и `quality_flags` не токенизируются без отдельного решения.
12. Конвенция календарных значений фиксирована: `hour_of_day_local` ∈ 0..23; `day_of_week_local` ∈ 0..6, понедельник = 0. Без явной конвенции реализации разойдутся (Пн=0 / Пн=1 / Вс=0) и golden-vectors не сойдутся.
13. `lifetime_first` — опциональный булев атрибут события: `true` только на самом раннем известном событии клиента (§25.1); отсутствие поля означает `false`.

## 6. КАНОНИЧЕСКАЯ СХЕМА ПРОФИЛЯ

Минимальная структура:

```json
{
  "client_id": "string",
  "profile_time_utc": "ISO-8601 UTC",
  "fields": {
    "region": "ALMATY",
    "employment": "EMPLOYED",
    "account_age_bucket": "bucket_12"
  },
  "source_meta": {},
  "quality_flags": []
}
```

Правила выбора:

1. Выбрать последний profile snapshot с `profile_time_utc <= T`.
2. Будущий snapshot запрещён.
3. Если snapshot отсутствует:
   - создать профиль с обязательными полями `MISSING`;
   - добавить `profile_snapshot_missing = true` (боковой quality-флаг, не токенизируется, если не включён в Feature Schema как отдельное поле профиля);
   - не брать первый будущий snapshot.
4. Все derived profile features рассчитываются только по данным `<= T`.

## 7. КАНОНИЧЕСКИЙ CLIENT_ID

Разные источники могут использовать:

- CIF;
- account_id;
- cardholder_id;
- CRM id;
- device-linked id;
- phone number;
- local customer id.

Preprocessing должен использовать единый `client_id`.

Правила:

1. Mapping источника к `client_id` должен быть versioned.
2. Неоднозначный mapping запрещено разрешать случайно.
3. Конфликтующие связи отправляются в quarantine.
4. PII не должно использоваться как публичный model identifier.
5. Изменение identity resolution требует новой preprocessing-версии.
6. Merge/split клиентов должен быть исторически корректным относительно T.

## 8. EVENT_ID, ИДЕМПОТЕНТНОСТЬ И LINEAGE

`event_id` должен быть стабильным: одинаковая исходная запись при повторном запуске даёт одинаковый `event_id`.

Рекомендуемый принцип:

```text
event_id =
stable_hash(
  source_system
  + source_record_id
  + normalized_event_type
  + event_timestamp
)
```

Если у источника нет надёжного `source_record_id`, используется бизнес-fingerprint из утверждённого набора полей.

Алгоритм `stable_hash` и каноничный пре-образ (порядок полей, кодирование, формат timestamp) фиксируются в §29.1 — без этого две реализации дадут разные `event_id`, разный tie-break и разный reservoir sample.

Не допускается использовать:

- случайный UUID при каждом запуске;
- текущее время;
- номер worker;
- порядок строки в файле без стабильного file identity.

Lineage должен позволять восстановить:

```text
canonical event
→ raw source record
→ source file / partition
→ source schema version
```

## 9. ДЕДУПЛИКАЦИЯ

Дедупликация выполняется до построения timeline.

### 9.1. Exact duplicates

Одинаковые:

- source system;
- source record ID;
- payload version.

Оставляется одна запись.

### 9.2. Business duplicates

Разные source record ID, но одинаковый бизнес-факт.

Используется versioned fingerprint.

Пример для транзакции:

```text
client_id
+ event_time
+ amount
+ currency
+ direction
+ merchant reference
```

### 9.3. Conflicting duplicates

Если одинаковый ключ содержит разные payload:

- применить source-specific correction policy;
- либо выбрать запись по version/update timestamp;
- либо отправить конфликт в quarantine.

Случайное `drop_duplicates(keep="first")` без правила запрещено.

### 9.4. Reversal и correction

Reversal не всегда является дубликатом.

Возможные варианты:

- отдельное событие `TRANSACTION_REVERSAL`;
- обновлённая версия исходного события;
- пара original/reversal.

Политика задаётся для каждого источника.

## 10. ТИПЫ СОБЫТИЙ И EVENT MAPPING

Каждая сырая запись должна быть преобразована в утверждённый `event_type`.

Примеры:

- `TRANSFER`;
- `PAYMENT`;
- `CASH_WITHDRAWAL`;
- `LOGIN`;
- `APP_SESSION`;
- `PUSH_NOTIFICATION`;
- `CARD_BLOCK`;
- `LOAN_PAYMENT`.

Mapping задаётся конфигурацией:

```text
source event code
→ canonical event_type
```

Правила:

1. Mapping versioned.
2. Неизвестный source code не создаёт новый event_type автоматически.
3. Неизвестный тип:
   - отправляется в quarantine;
   - увеличивает `unknown_event_type_rate`.
4. Значение `OTHER` допускается только как утверждённая бизнес-категория.
5. Технический source message не обязан становиться модельным событием.

## 11. FEATURE SCHEMA

Feature Schema является общим контрактом preprocessing и токенайзера.

`event_type` является top-level атрибутом canonical event и не включается в список `fields`.

Для каждого field задаются:

- тип;
- обязательность;
- допустимые значения;
- порядок;
- MISSING policy;
- numeric/bucketization policy;
- multivalue policy;
- max_values_per_field;
- high-cardinality policy;
- text policy;
- field priority;
- model inclusion flag;
- `vocabulary_policy`.

Допустимые `vocabulary_policy`:

- `closed_set`;
- `bucket_closed_set`;
- `frequency_pruned`;
- `excluded`.

Пример:

```yaml
TRANSFER:
  event_type: TRANSFER
  fields:
    - name: amount_base_bucket
      type: bucket
      required: true
      vocabulary_policy: bucket_closed_set
      domain: [bucket_0, bucket_1, ..., bucket_63, MISSING]
      priority: 1
    - name: currency
      type: categorical
      required: true
      vocabulary_policy: closed_set
      domain: [KZT, USD, EUR, MISSING]
      priority: 2
    - name: merchant_category
      type: categorical
      required: false
      vocabulary_policy: frequency_pruned
      priority: 3
```

### 11.1. Bucket-field whitelist

Для каждого bucket-поля preprocessing публикует полный domain независимо от фактической частоты бакета в TRAIN.

Токенайзер обязан включить все values bucket domain и не применять к ним `min_count/RARE`.

Feature Schema ничего не изменяет сама.
Её применяет preprocessing-код.

### 11.2. Профильные поля

Feature Schema покрывает не только события: отдельная секция `PROFILE` описывает поля профиля (§6) теми же атрибутами, включая `vocabulary_policy`:

- бакетированные профильные поля (`account_age_bucket`, ...) → `bucket_closed_set`;
- справочные категории (`region`, `employment`, ...) → `closed_set`;
- open-set профильных полей следует избегать; при необходимости — `frequency_pruned`.

Без этого правила токенайзера «`MISSING` для всех полей» и исключения из `min_count` неприменимы к профилю.

## 12. НОРМАЛИЗАЦИЯ TIMESTAMP И ЧАСОВЫХ ПОЯСОВ

Каждая запись должна иметь event time и утверждённую timezone policy.

Preprocessing хранит два разных представления:

1. `timestamp_utc` — абсолютный instant для сортировки, cutoff и lineage.
2. `calendar_timezone` + локальные признаки — для поведенческих hour/day features.

Правила:

1. Timestamp парсится строго по Source Contract.
2. Instant переводится в UTC.
3. Для календарных признаков instant конвертируется в бизнес-локальную IANA timezone.
4. `hour_of_day_local` и `day_of_week_local` нельзя вычислять напрямую из UTC, если бизнес-смысл локальный.
5. IANA timezone предпочтительнее фиксированного offset, потому что хранит исторические изменения.
6. Наивный timestamp без timezone обрабатывается только по source-specific policy.
7. Неизвестная timezone:
   - запись quarantined;
   - либо применяется явно утверждённый source default.
8. Processing time не заменяет отсутствующий event time.
9. Timestamp из будущего относительно clock-skew policy маркируется.
10. Выходной instant — ISO-8601 UTC или epoch утверждённой точности.

### 12.1. Политика для Казахстана

Для событий на территории Казахстана календарные признаки рассчитываются по утверждённой IANA timezone события/региона.

С 1 марта 2024 года Казахстан использует единый UTC+05:00.

Основание: постановление Правительства Республики Казахстан от 19 января 2024 года № 20.

Для исторических событий до этой даты запрещено ретроактивно применять фиксированный UTC+05 ко всей стране: необходимо использовать исторические правила IANA timezone соответствующего региона.

### 12.2. Event time и processing time

Для timeline используется `event_time`.

`processing_time` используется только для lineage, watermark и late-arriving monitoring.

## 13. ДЕТЕРМИНИРОВАННЫЙ ПОРЯДОК СОБЫТИЙ

События клиента сортируются по:

1. `timestamp_utc`;
2. `source_priority`;
3. `source_record_id` или стабильный `event_id`.

Это создаёт `ordering_key`.

`source_priority` задаётся в Source Contract каждого источника (§4) и версионируется через `source_contract_version`; это детерминированный ранг, не зависящий от worker, файла или порядка завершения task.

Если два события имеют одинаковый timestamp, порядок не должен зависеть от:

- порядка файла;
- worker;
- времени завершения task;
- случайного UUID.

Одинаковые данные всегда должны давать одинаковый timeline.

## 14. CUTOFF TIME T И ЗАЩИТА ОТ LEAKAGE

Главное правило:

```text
в prepared_events входят только события timestamp_utc <= T
```

Также:

```text
profile_time_utc <= T
```

Cutoff применяется до:

- выбора профиля;
- расчёта агрегатов;
- life-long features;
- временных дельт;
- sessionization, если сессия пересекает T;
- bucketization примера;
- передачи токенайзеру.

### 14.1. Сессия, пересекающая T

Если app session началась до T, но содержит действия после T:

- действия после T исключаются;
- session summary пересчитывается только по доступной части;
- нельзя использовать будущую длительность сессии.

### 14.2. Training target

Target может находиться после T, но не должен попадать во входные признаки.

Target horizon и observation window задаются отдельно.

## 15. ПРОПУСКИ И НЕВАЛИДНЫЕ ЗНАЧЕНИЯ

### 15.1. MISSING

Если поле применимо, но значение отсутствует:

```text
field = MISSING
```

Если поле неприменимо:

```text
поле не добавляется
```

### 15.2. Источники пропусков

К MISSING приводятся:

- `NULL`;
- `None`;
- пустая строка после trim;
- утверждённые source placeholders;
- отсутствующий обязательный ключ;
- невалидное значение, если policy предписывает MISSING.

### 15.3. MISSING не равен нулю

Запрещено без бизнес-основания:

```text
missing amount → 0
missing duration → 0
```

Ноль может быть реальным значением.

### 15.4. MISSING не равен [UNK]

`MISSING` создаётся preprocessing.

`[UNK]` применяется токенайзером после freeze Vocabulary.

## 16. НОРМАЛИЗАЦИЯ КАТЕГОРИЙ

Для категориальных полей задаётся canonicalization policy.

Типовые операции:

- trim;
- Unicode normalization;
- case normalization;
- alias mapping;
- transliteration только по утверждённому правилу;
- справочное mapping;
- удаление технических префиксов;
- объединение синонимов.

Пример:

```text
"kzt", "тенге", "398"
→ "KZT"
```

### 16.1. Closed-set category

Поле имеет закрытый набор.

Пример:

```text
direction ∈ {IN, OUT}
```

Невалидное значение:

- переводится в `MISSING`;
- либо вызывает schema violation;
- не создаёт новую категорию автоматически.

### 16.2. Open-set category

Поле допускает новые значения.

Пример:

- merchant category extension;
- device model;
- template ID.

Preprocessing нормализует строку, но решение `RARE/[UNK]` принимает токенайзер.

### 16.3. OTHER

`OTHER` разрешён только если это утверждённая категория со стабильным смыслом.

`OTHER` не должен использоваться как универсальная замена любых ошибок.

## 17. ОБРАБОТКА ЧИСЛОВЫХ ПОЛЕЙ

Для каждого числового поля Feature Schema задаёт:

- numeric type;
- signed/unsigned;
- допустимый диапазон;
- единицу измерения;
- parsing locale;
- missing policy;
- bucketization method;
- clipping policy.

### 17.1. Parsing

Перед преобразованием удаляются утверждённые:

- разделители тысяч;
- пробелы;
- locale decimal separators;
- единицы измерения, если это разрешено контрактом.

### 17.2. Невалидные числа

К невалидным относятся:

- непарсибельная строка;
- `NaN`;
- `+Inf`;
- `-Inf`;
- пустая строка;
- число вне business-valid range.

Базовое правило:

```text
invalid numeric
→ field-specific MISSING
```

Запрещено:

- заменять на 0;
- отправлять в `[UNK]`;
- создавать новый бакет;
- падать на обычном production inference.

Мониторинг:

- `numeric_parse_error_rate`;
- `numeric_nan_rate`;
- `numeric_business_range_error_rate`.

### 17.3. Отрицательные значения

Отрицательное значение может быть:

- валидным, например signed balance delta;
- невалидным, например age.

Правило задаётся отдельно для каждого поля.

## 18. FX-НОРМАЛИЗАЦИЯ СУММ

Основная стратегия:

```text
исходная сумма
→ исторический FX-rate на timestamp
→ базовая валюта KZT
→ bucketization
```

Исходная `currency` сохраняется отдельным признаком.

### 18.1. Источник курса

FX source должен быть:

- утверждён;
- versioned;
- исторически воспроизводим;
- доступен по дате/времени;
- иметь lineage.

### 18.2. Fallback

Если точного курса нет:

1. Использовать последний доступный курс не позднее event timestamp.
2. Максимальная давность определяется `fx_max_staleness`.
3. Базовое значение: 3 календарных дня.
4. Будущий курс использовать запрещено.
5. Если допустимого прошлого курса нет:
   - `amount_base_bucket = MISSING`;
   - исходная currency сохраняется;
   - событие не удаляется;
   - увеличивается `fx_missing_rate`.

### 18.3. Особые денежные поля

Не все суммы должны использовать одинаковые edges.

Отдельные поля:

- transaction amount;
- balance;
- credit limit;
- salary;
- overdue amount.

Каждое поле имеет собственные bucket_edges.

Соглашение имён bucket-полей: FX-нормализованные денежные суммы → `<field>_base_bucket` (значение в KZT после §18); прочие числовые поля → `<field>_bucket`. Какие поля проходят FX-нормализацию, фиксируется в Feature Schema per-field.

## 19. BUCKETIZATION

Numeric bucketization полностью принадлежит preprocessing.

`bucket_edges` рассчитываются только на TRAIN.
Validation, Test и Production используют frozen edges.

### 19.1. Исключение MISSING

MISSING и невалидные значения не участвуют в fit bucket_edges.

### 19.2. Метод

Допустимые методы:

- quantile;
- equal-width;
- log-space;
- business-defined edges.

Метод задаётся по полю.

Количество бакетов — гиперпараметр по полю. Стартовые варианты: 16, 32, 64, 128. Выбор выполняется по Validation.

### 19.3. Повторяющиеся quantile edges

Если edges совпадают:

- дубли удаляются;
- фактическое число бакетов фиксируется;
- domain публикуется в `bucket_field_domains`.

### 19.4. Clipping вне TRAIN-диапазона

```text
value < min_train_edge → lowest_bucket
value > max_train_edge → highest_bucket
```

Новый bucket не создаётся.

### 19.5. Выход preprocessing

После transform поле содержит только:

```text
bucket_0 … bucket_N
MISSING
```

Для каждого bucket-поля сохраняются:

- edges;
- labels;
- closed-set domain;
- MISSING token;
- decode interval metadata;
- low/high clip metrics.

Tokenizer получает bucket label как категорию и не применяет edges повторно.

### 19.6. Детерминированный sample

Если используется sample:

- deterministic reservoir sampling;
- seed от stable record hash и global seed (алгоритм хэша и пре-образ — §29.1);
- результат не зависит от worker count.

## 20. SESSIONIZATION APP-ЛОГОВ

Сырые app-логи часто представлены множеством строк:

```text
screen_open
button_click
screen_open
form_submit
```

Preprocessing может объединять их в `APP_SESSION`.

### 20.1. Базовое правило сессии

Новая сессия начинается, если:

- клиент изменился;
- разрыв между действиями больше `session_gap`;
- присутствует явный session boundary;
- приложение перезапущено по утверждённому сигналу.

Стартовое значение:

```text
session_gap = 30 минут
```

Это конфигурация, а не универсальный стандарт.

### 20.2. Поля APP_SESSION

Пример:

- session_start;
- duration_bucket;
- screens_count_bucket;
- top_intent;
- first_screen;
- last_screen;
- selected screens;
- channel/device category.

### 20.3. Ограничения

- действия после T не используются;
- duration не смотрит в будущее;
- порядок screens детерминирован;
- действует `max_values_per_field`;
- raw click ID не включается;
- sessionization_config versioned.

## 21. МНОГОЗНАЧНЫЕ ПОЛЯ

Примеры:

- products;
- screens;
- channels;
- active cards;
- intents.

Правила:

1. Определить, значим ли порядок.
2. Если порядок незначим — сортировать.
3. Если порядок значим — сохранить хронологию.
4. Применить `max_values_per_field` (базовое значение: 8; конфигурация по полю).
5. При обрезке использовать утверждённый priority.
6. При необходимости добавить `count_bucket`.
7. Не создавать гигантское событие.

Пример:

```text
screens = первые N + последние N + top_intent
```

## 22. ВЫСОКОКАРДИНАЛЬНЫЕ ПОЛЯ

Примеры:

- transaction_id;
- session_id;
- device_id;
- merchant_id;
- raw template_id.

Стратегии:

1. исключить технический ID;
2. заменить категорией;
3. сохранить утверждённые частые значения;
4. hashing как отдельная версия;
5. передать нормализованное значение токенайзеру для `min_count/RARE`.

Preprocessing не назначает `RARE`.
Он только применяет high-cardinality policy и формирует стабильное значение поля.

## 23. ТЕКСТОВЫЕ ПОЛЯ

Базовая архитектура запрещает необработанный свободный текст.

Запрещено передавать напрямую:

- payment description;
- support messages;
- notification body;
- raw merchant name;
- comments.

Разрешено:

- утверждённая категория;
- безопасный классификатор intent;
- справочное нормализованное значение;
- заранее рассчитанный признак;
- отдельная будущая BPE-архитектура.

Preprocessing должен:

- удалить PII;
- применить text policy;
- не передавать текст токенайзеру.

## 24. LIFE-LONG ПРИЗНАКИ ПРОФИЛЯ

Truncation истории не должен удалять сведения о возрасте клиента и давних фактах.

Минимальный набор:

- account_age;
- first_seen_age;
- first_topup_age;
- lifetime_event_count;
- lifetime_transaction_count;
- lifetime_product_count.

Порядок:

```text
raw lifetime value
→ validation
→ bucketization
→ profile field
```

Все признаки рассчитываются на T.

Запрещено использовать:

- полную будущую lifetime history;
- текущий production age для исторического TRAIN-примера.

## 25. ВРЕМЕННЫЕ МЕТАДАННЫЕ ДЛЯ TIME EMBEDDING

Preprocessing хранит UTC instant и рассчитывает локальные календарные признаки.

Обязательный выход для каждого события:

- `timestamp_utc`;
- `calendar_timezone`;
- `hour_of_day_local`;
- `day_of_week_local`.

Конвенция значений — §5, правило 12 (час 0..23; день недели 0..6, понедельник = 0).

### 25.1. Владение delta_from_previous_event

Финальная model-input `delta_from_previous_event` не рассчитывается preprocessing как неизменяемое значение.

Причина: model-input pipeline выполняет truncation и знает итоговое окно.

Правильный порядок:

```text
preprocessing: timestamp_utc + calendar features
→ tokenizer/model-input: выбрать surviving window
→ пересчитать delta между surviving events
→ первому event окна назначить WINDOW_START
```

`FIRST_EVENT` используется только для истинно первого lifetime event, если перед ним не было отброшенной истории.

Чтобы токенайзер мог отличить `FIRST_EVENT` от `WINDOW_START` без загрузки полной истории, preprocessing передаёт на самом раннем известном событии клиента флаг `lifetime_first = true` (см. tokenizer §10.1). При загрузке полного ≤T timeline флаг избыточен, но при загрузке только недавнего среза он обязателен.

Preprocessing может считать full-timeline delta только для QA/monitoring, но это поле не должно использоваться моделью после truncation без пересчёта.

### 25.2. Time delta edges

Если delta кодируется бакетами:

- `time_delta_edges` fit только на TRAIN;
- edges публикуются как versioned time artifact;
- transform применяется после truncation в model-input pipeline;
- numeric event bucketization и time-delta bucketization являются разными компонентами;
- `WINDOW_START` и `FIRST_EVENT` — зарезервированные значения delta-канала вне обычных `time_delta_edges` (представление — tokenizer §10.2).

### 25.3. Связь с событием

Calendar time features и timestamp связываются с `event_id`.

Time metadata не токенизируется как обычные key/value fields и не расходует `max_tokens_per_event`.

## 26. ПОСТРОЕНИЕ ЕДИНОГО TIMELINE

После подготовки все события клиента объединяются в один timeline.

Пример:

```text
09:00 LOGIN
09:03 APP_SESSION
09:10 TRANSFER
11:20 PUSH_NOTIFICATION
```

Правила:

1. Используется canonical `event_type`.
2. Используется `timestamp_utc`.
3. Применяется детерминированный tie-break.
4. Дубликаты удалены до timeline.
5. Все события `<= T`.
6. Source-specific metadata не меняет модельный порядок без утверждённого правила.
7. Timeline сохраняется полностью; truncation выполняется позже model-input pipeline.

## 27. BUILD PHASE

BUILD PHASE выполняется только на TRAIN.

Шаги:

1. Зафиксировать Source Contracts.
2. Зафиксировать identity и event mappings.
3. Зафиксировать Feature Schema без event_type в fields.
4. Зафиксировать closed-set и bucket domains.
5. Зафиксировать cutoff и timezone policies.
6. Зафиксировать dedup и sessionization configs.
7. Зафиксировать category mappings.
8. Зафиксировать FX source и `fx_max_staleness`.
9. Зафиксировать numeric validation rules.
10. Отобрать TRAIN без leakage.
11. Выполнить deterministic sampling.
12. Рассчитать numeric bucket_edges.
13. Рассчитать time_delta_edges.
14. Опубликовать bucket_field_domains.
15. Рассчитать TRAIN baselines.
16. Проверить single/multi-worker equality.
17. Канонически сериализовать artifacts.
18. Рассчитать `preprocessing_state_sha256`.
19. Заморозить и зарегистрировать версию.

BUILD PHASE не выполняется на отдельном примере, batch, Validation, Test или inference.

## 28. ENCODE PHASE

1. Загрузить и проверить `preprocessing_state_sha256`.
2. Определить cutoff T.
3. Применить Source Contract.
4. Разрешить canonical client_id.
5. Нормализовать timestamp в UTC.
6. Определить calendar_timezone.
7. Отобрать records `<= T`.
8. Выполнить deduplication.
9. Построить top-level canonical event_type.
10. Выполнить sessionization.
11. Применить Feature Schema к fields без event_type.
12. Подставить MISSING.
13. Нормализовать категории.
14. Провалидировать числа.
15. Применить FX normalization.
16. Применить frozen numeric bucket_edges и clipping.
17. Проверить bucket value против bucket_field_domain.
18. Ограничить multivalue fields.
19. Выбрать profile snapshot на T.
20. Рассчитать life-long features.
21. Построить timeline.
22. Рассчитать local hour/day.
23. Не фиксировать final delta_prev до model window.
24. Выполнить финальную schema validation.
25. Передать prepared profile/events, domains и metadata токенайзеру.
26. Записать metrics и lineage.

## 29. ДЕТЕРМИНИЗМ ПРИ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКЕ

Одинаковые входные данные и config должны давать байт-в-байт одинаковые preprocessing-артефакты и prepared dataset независимо от workers.

Правила:

1. Файлы сортируются по canonical path или stable file ID.
2. Партиции сортируются.
3. Merge результатов выполняется в строгом порядке.
4. Stable hashes не зависят от worker ID.
5. Reservoir sample не зависит от числа workers.
6. Bucket edges не зависят от порядка завершения tasks.
7. Event IDs не используют случайность.
8. Timeline tie-break детерминирован.
9. Каноническая сериализация фиксирована (см. §29.1): кодировка, порядок ключей и формат чисел.
10. Single-worker и multi-worker outputs сравниваются.

Несовпадение считается блокирующей ошибкой.

### 29.1. Каноническая сериализация и stable hash

Content hash и сравнение single/multi-worker требуют байт-идентичной сериализации между платформами, языками и версиями библиотек. Обязательно фиксируются:

1. Кодировка — UTF-8 без BOM.
2. Порядок ключей объектов — лексикографический по UTF-8 byte order.
3. Пробелы и переводы строк — компактный канонический вид без незначащих пробелов.
4. Формат float — единый утверждённый вариант (зафиксировать в `serialization_config` и версионировать), один из:
   - shortest round-trip repr IEEE-754 double;
   - фиксированное число знаков после запятой на поле;
   - hex IEEE-754 (побитовая запись).
5. `-0.0` нормализуется в `0.0`.
6. `NaN`, `+Inf`, `-Inf` запрещены в `bucket_edges`, `time_delta_edges` и любом хэшируемом артефакте; их появление — блокирующая ошибка BUILD.
7. Целые и булевы значения сериализуются без плавающей точки.
8. Выбранный numeric format входит в `preprocessing_state_sha256`; его изменение требует новой `preprocessing_version`.

Stable hash policy — единые правила для `event_id`, business fingerprint и reservoir seed (§8, §9.2, §19.6):

9. Алгоритм — один именованный хэш на весь pipeline: `SHA-256`. Встроенный `hash()` языка запрещён.
10. Каноничный пре-образ: поля в фиксированном порядке (для `event_id` — порядок §8), каждое поле как UTF-8 с length-prefix (только length-prefix; символ-разделитель не допускается — «не может встретиться в данных» недоказуемо); timestamp — целое число epoch микросекунд в десятичной записи, без локали и без float-секунд.
11. `event_id` — первые 128 бит (32 hex-символа) SHA-256 пре-образа.
12. Reservoir seed — первые 8 байт SHA-256 record key как uint64 big-endian, комбинируются с `global_seed` утверждённой формулой.
13. Всё вышеперечисленное фиксируется как `hash_policy`, версионируется (`hash_policy_version`), входит в `preprocessing_state_sha256`; изменение требует новой `preprocessing_version`.

### 29.2. Golden-vector conformance set

Сравнение single/multi-worker (п.10) гарантирует идентичность лишь одной реализации при разном числе workers. Для идентичности **между реализациями** (другая машина, версия библиотек, язык/движок) обязателен замороженный golden-vector conformance set.

Состав:

1. `golden_input` — фиксированный небольшой набор **сырых** записей, покрывающий краевые случаи: `MISSING`, сумму в иностранной валюте (FX→KZT), число вне TRAIN-диапазона (clipping), два события с одинаковым `timestamp_utc` (tie-break), событие на границе `T`, отсутствующий profile snapshot, дубликат, а также случаи, доживающие до токенайзера: open-set значение, встречавшееся в TRAIN реже `min_count` (→ `RARE`), open-set значение, отсутствующее в TRAIN (→ `[UNK]`), и клиент с историей длиннее model window (→ `WINDOW_START`).
2. `golden_expected` — заморожённый ожидаемый выход: `prepared_profile`, `prepared_events`, `bucket_field_domains` и `preprocessing_state_sha256`.

Правила:

1. Значения `golden_expected` заполняются первым эталонным прогоном ENCODE на утверждённых артефактах; вручную не составляются.
2. Любая реализация, получив `golden_input` и утверждённые артефакты, обязана воспроизвести `golden_expected` байт-в-байт.
3. Несовпадение — блокирующая ошибка релиза.
4. Golden set версионируется (`golden_vectors_version`); изменение эталонных артефактов требует новой версии.
5. Golden set входит в обязательные артефакты (§31) и прогоняется в CI перед каждым релизом.
6. `golden_input` синтетический, без реальных PII.

## 30. ВЕРСИОНИРОВАНИЕ И CONTENT HASH

Единый комплект:

- source_contract_version;
- identity_mapping_version;
- event_mapping_version;
- feature_schema_version;
- closed_set_domains_version;
- bucket_field_domains_version;
- category_mapping_version;
- timestamp_policy_version;
- calendar_timezone_policy_version;
- dedup_policy_version;
- sessionization_version;
- fx_normalization_version;
- fx_max_staleness;
- bucket_edges_version;
- time_delta_edges_version;
- text_policy_version;
- hash_policy_version;
- golden_vectors_version;
- preprocessing_version.

`preprocessing_state_sha256` рассчитывается по канонически сериализованному состоянию, включая:

- Source Contracts;
- mappings;
- Feature Schema;
- closed-set domains;
- bucket field domains;
- timestamp/calendar timezone policies;
- dedup/sessionization configs;
- FX config;
- bucket edges;
- time delta edges;
- numeric rules;
- text/cutoff policies;
- serialization config.

Сериализация выполняется строго по §29.1 (единый формат float для `bucket_edges`/`time_delta_edges`, запрет `NaN/Inf`, нормализация `-0.0 → 0.0`).

Hash проверяется при загрузке.
Несовпадение блокирует обработку.

## 31. ОБЯЗАТЕЛЬНЫЕ АРТЕФАКТЫ

Необходимо сохранять:

- source_contracts;
- identity_mapping;
- event_mapping;
- feature_schema;
- closed_set_domains;
- bucket_field_domains;
- category_mappings;
- dedup_config;
- sessionization_config;
- timestamp_policy;
- calendar_timezone_policy;
- cutoff_policy;
- fx_normalization_config;
- fx_max_staleness;
- bucket_edges;
- bucket_metadata;
- time_delta_edges;
- numeric_validation_rules;
- high_cardinality_policy;
- text_policy;
- field_priorities;
- max_values_per_field;
- TRAIN baselines;
- deterministic_build_config;
- hash_policy (§29.1);
- golden_vectors (input + expected, §29.2);
- sampling_algorithm;
- global_seed;
- preprocessing_state_sha256;
- code commit/hash;
- TRAIN dataset identifier;
- build timestamp.

## 32. OUTPUT DATASET

### 32.1. prepared_profile

```json
{
  "client_id": "123",
  "profile_time_utc": "2026-01-01T00:00:00Z",
  "fields": {
    "region": "ALMATY",
    "employment": "EMPLOYED",
    "account_age_bucket": "bucket_12"
  }
}
```

### 32.2. prepared_events

```json
[
  {
    "client_id": "123",
    "event_id": "evt_1",
    "event_type": "TRANSFER",
    "timestamp_utc": "2026-01-15T09:30:00Z",
    "calendar_timezone": "Asia/Almaty",
    "ordering_key": "...",
    "fields": {
      "amount_base_bucket": "bucket_17",
      "currency": "KZT",
      "channel": "MOBILE"
    },
    "calendar_time_features": {
      "hour_of_day_local": 14,
      "day_of_week_local": 3
    }
  }
]
```

`event_type` находится только top-level.

`delta_from_previous_event` не передаётся как финальное model-input значение до выбора окна.

### 32.3. Метаданные

```json
{
  "cutoff_time": "2026-01-31T23:59:59Z",
  "preprocessing_version": "...",
  "preprocessing_state_sha256": "...",
  "feature_schema_version": "...",
  "bucket_field_domains_version": "...",
  "bucket_edges_version": "...",
  "calendar_timezone_policy_version": "..."
}
```

## 33. PRODUCTION-МОНИТОРИНГ

Пороговые значения уточняются по baseline банка.

### 33.1. Cutoff violation rate

- любое значение > 0 — critical.

### 33.2. Schema violation rate

- > 0.1% — warning;
- > 1% — critical.

### 33.3. Unknown event type rate

- > 0% — warning;
- > 0.1% — critical.

### 33.4. Bucket domain violation rate

- любое значение > 0 — critical.

### 33.5. Dedup conflict rate

- рост > 2x baseline — warning;
- рост > 5x — critical.

### 33.6. MISSING rate

По каждому полю:

- рост > 2x — warning;
- рост > 5x — critical.

### 33.7. Numeric parse error rate

- <= 0.01% — normal;
- > 0.01% — warning;
- > 0.1% — critical.

### 33.8. Numeric clip rate

- рост > 2x — warning;
- рост > 5x — critical.

### 33.9. FX missing rate

- <= 0.1% — normal;
- > 0.1% — warning;
- > 1% — critical.

### 33.10. Timestamp/timezone error rate

- > 0% — warning;
- > 0.1% — critical.

### 33.11. Calendar timezone fallback rate

Контролируется по source/region.
Резкий рост означает ухудшение timezone metadata.

### 33.12. Late-arriving rate

Сравнивается с baseline по источнику.

### 33.13. Session anomaly rate

Контролируются negative/excessive duration, empty session и too many values.

### 33.14. Profile missing rate

Контролируется по сегментам клиентов.

## 34. QUARANTINE И ERROR HANDLING

Запись отправляется в quarantine, если:

- невозможно определить client_id;
- невозможно определить event_type;
- отсутствует обязательный event_time;
- timezone неизвестен и нет source policy;
- нарушен Source Contract;
- конфликтующий duplicate не разрешён;
- schema version несовместима;
- content hash не совпадает.

Quarantine record должен содержать:

- raw reference;
- reason code;
- source;
- processing time;
- pipeline version;
- recoverability flag.

Запрещено молча удалять проблемные записи без метрики и lineage.

## 35. QA-ЧЕК-ЛИСТ

### A. EVENT CONTRACT

1. `event_type` находится top-level.
2. `event_type` отсутствует внутри fields.
3. Fields соответствуют Feature Schema.
4. Bucket fields имеют `bucket_closed_set`.
5. Полный bucket domain опубликован для токенайзера.

### B. TIME И CUTOFF

6. timestamp хранится в UTC.
7. Calendar timezone является IANA timezone.
8. Hour/day рассчитаны в local timezone, а не напрямую из UTC.
9. Для Казахстана после 2024-03-01 применяется UTC+05 через IANA rules.
10. Для истории до 2024-03-01 используются historical regional IANA rules.
11. Все events имеют timestamp <= T.
12. Profile snapshot <= T.
13. Final delta_prev не фиксируется до truncation.
14. Timestamp/event_id (и `lifetime_first`, где применимо) передаются для window-relative delta recomputation.

### C. NUMERIC, FX И BUCKETS

15. Parsing, NaN/Inf и MISSING обработаны до токенайзера.
16. FX normalization выполнена до bucketization.
17. Future FX запрещён.
18. Clipping выполнен preprocessing.
19. Bucket value входит в configured domain.
20. Bucket metadata передаётся только для decode.
21. Tokenizer не должен повторно применять edges.
22. Все bucket values whitelist независимо от частоты.
23. Хвостовой bucket не должен стать RARE.

### D. IDENTITY, DEDUP И TIMELINE

24. client_id/event_id детерминированы.
25. Lineage восстановим.
26. Exact/business duplicates обработаны.
27. Conflicts не разрешены случайным keep-first.
28. Timeline tie-break детерминирован.

### E. SESSION, PROFILE И TEXT

29. Sessionization не использует future actions.
30. max_values_per_field соблюдается.
31. Profile выбран на T.
32. Life-long признаки рассчитаны на T.
33. Free text отсутствует.

### F. BUILD/ENCODE И INTEGRITY

34. BUILD выполняется только на TRAIN.
35. ENCODE использует frozen artifacts.
36. Single/multi-worker outputs совпадают байт-в-байт.
37. closed_set_domains и bucket_field_domains versioned.
38. preprocessing_state_sha256 рассчитан.
39. Hash проверяется при загрузке.
40. Output metadata содержит все версии.

### G. INTERFACE С ТОКЕНИЗАТОРОМ

41. prepared events не требуют numeric/FX/clipping operations.
42. Tokenizer получает top-level event_type.
43. Tokenizer получает local calendar features.
44. Tokenizer получает timestamps для post-truncation delta.
45. Tokenizer получает bucket domains для min_count exemption.
46. Tokenizer не получает raw numeric values.
47. Tokenizer не пересчитывает preprocessing monitoring.

### H. GOLDEN-VECTORS

48. `golden_input`/`golden_expected` существуют и версионированы.
49. Эталонная реализация воспроизводит `golden_expected` байт-в-байт.
50. Несовпадение golden-vector блокирует релиз.

## 36. РЕКОМЕНДУЕМАЯ СТРУКТУРА КОДА

1. `SourceReader`
   - чтение источников;
   - Source Contract;
   - schema validation.

2. `IdentityResolver`
   - canonical client_id;
   - historical mapping.

3. `TimestampNormalizer`
   - parsing;
   - timezone;
   - UTC.

4. `Deduplicator`
   - exact;
   - business fingerprint;
   - conflict policy.

5. `EventMapper`
   - raw code → canonical event_type;
   - event construction.

6. `Sessionizer`
   - app sessions;
   - session summaries.

7. `FeatureSchemaRegistry`
   - fields;
   - types;
   - required;
   - priorities;
   - policies.

8. `CategoryNormalizer`
   - aliases;
   - canonical categories.

9. `NumericValidator`
   - parsing;
   - bounds;
   - MISSING.

10. `FXNormalizer`
    - historical rates;
    - staleness;
    - fallback.

11. `Bucketizer`
    - deterministic fit;
    - transform;
    - clipping.

12. `ProfileBuilder`
    - snapshot selection;
    - life-long features.

13. `TimeFeatureBuilder`
    - delta_prev;
    - hour;
    - day_of_week.

14. `TimelineBuilder`
    - sorting;
    - tie-break;
    - cutoff.

15. `DeterministicSampler`
    - reservoir sample;
    - stable hash.

16. `ArtifactHasher`
    - canonical serialization;
    - SHA-256.

17. `DataQualityMonitor`
    - metrics;
    - thresholds;
    - quarantine.

18. `PreprocessingPipeline`
    - orchestration;
    - BUILD/ENCODE separation.

## 37. КАНОНИЧЕСКИЙ ПОРЯДОК

### 37.1. BUILD PHASE

```text
Source Contracts
→ identity/event mapping
→ Feature Schema without event_type in fields
→ closed-set and bucket domains
→ timestamp + calendar timezone policy
→ cutoff/dedup/sessionization policies
→ category/numeric/FX rules
→ deterministic TRAIN sample
→ numeric bucket_edges
→ time_delta_edges
→ bucket_field_domains
→ baselines
→ single/multi-worker comparison
→ canonical serialization
→ preprocessing_state_sha256
→ freeze
```

### 37.2. ENCODE PHASE

```text
raw records
→ Source Contract
→ client_id
→ timestamp UTC + calendar timezone
→ cutoff T
→ dedup
→ top-level event_type
→ sessionization
→ Feature Schema fields
→ MISSING
→ categories
→ numeric validation
→ FX normalization
→ frozen bucketization + clipping
→ bucket domain validation
→ profile on T
→ life-long features
→ timeline
→ local hour/day
→ prepared profile/events
→ tokenizer
→ tokenizer truncates window
→ tokenizer recomputes delta_prev
```

## 38. КЛЮЧЕВЫЕ ПРАВИЛА

1. Preprocessing и tokenizer имеют разное владение.
2. Numeric parsing, FX, bucketization и clipping выполняются только preprocessing.
3. Tokenizer получает bucket labels как closed-set категории.
4. Bucket edges передаются tokenizer только для decode/observability.
5. Все bucket values whitelist и не зависят от min_count.
6. `event_type` хранится top-level.
7. `event_type` не дублируется в fields.
8. Все events имеют timestamp <= T.
9. UTC хранит instant и определяет ordering.
10. Hour/day считаются в business-local IANA timezone.
11. Для Казахстана с 2024-03-01 действует единый UTC+05.
12. Исторические события используют исторические IANA rules.
13. Final delta_prev вычисляется после tokenizer truncation.
14. Первое event окна получает WINDOW_START в model-input pipeline.
15. Feature Schema обязательна.
16. MISSING создаётся preprocessing.
17. Невалидные числа не превращаются в [UNK].
18. Sessionization не использует future actions.
19. Life-long features рассчитаны на T.
20. BUILD и ENCODE разделены.
21. Параллельный BUILD детерминирован.
22. preprocessing_state_sha256 обязателен.
23. Ошибочные records quarantined, а не удаляются молча.
24. TRAIN и Production используют frozen preprocessing state.
25. Выход preprocessing полностью соответствует tokenizer contract.

КОНЕЦ ДОКУМЕНТА
