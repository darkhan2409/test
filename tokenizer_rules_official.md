# Регламент построения и использования токенайзера

*Для банковской событийной модели*

- **Статус:** официальный регламент
- **Связанный документ:** `preprocessing_rules_official`

## Назначение документа

определить единые правила преобразования подготовленных банковских данных
в последовательности token_id и сопутствующие модельные признаки.

Все правила встроены в основные разделы.
Приложения с переопределяющими правилами отсутствуют.
При реализации необходимо руководствоваться только настоящим документом.

## 1. ГРАНИЦЫ ОТВЕТСТВЕННОСТИ

Токенайзер работает только с результатом утверждённого preprocessing pipeline.

**На вход поступают данные, которые уже:**

- очищены и дедуплицированы;
- ограничены cutoff time T;
- приведены к каноническим профилю и событиям;
- имеют top-level `event_type`;
- имеют нормализованные категориальные значения;
- содержат явный `MISSING`;
- содержат уже рассчитанные bucket-поля;
- содержат локальные календарные признаки времени;
- не содержат сырых числовых значений, требующих parsing или bucketization;
- не содержат необработанного FX;
- не содержат запрещённого свободного текста;
- соответствуют Feature Schema и версии preprocessing.

**Токенайзер отвечает за:**

- добавление `[USR]` и `[EVT]`;
- явную эмиссию `event_type` из top-level поля события;
- формирование key/value-токенов;
- построение Vocabulary на TRAIN;
- применение `min_count` только к разрешённым open-set полям;
- преобразование строковых токенов в `token_id`;
- обработку `RARE` и `[UNK]`;
- encode/decode;
- сборку model-input последовательности;
- truncation по целым событиям;
- пересчёт window-relative time delta;
- padding, attention mask и masking через отдельные model-input компоненты.

**Токенайзер не должен:**

- очищать сырые данные;
- парсить числовые поля;
- выполнять FX-нормализацию;
- рассчитывать или применять bucket_edges при кодировании;
- выполнять clipping;
- подставлять MISSING в сырые поля;
- sessionize app-логи;
- строить профиль на T;
- выбирать события по cutoff T;
- вычислять локальный час или день недели из UTC;
- менять Vocabulary в production;
- автоматически включать новые токены в старую модель.

Если токенайзер получает данные, требующие перечисленных preprocessing-операций, вход считается нарушившим контракт.

## 2. ОБЯЗАТЕЛЬНЫЙ CUTOFF TIME T

Cutoff time T применяется preprocessing pipeline до передачи данных токенайзеру.

Токенайзер не выбирает события по времени заново, но обязан валидировать входной контракт:

```text
event.timestamp_utc <= T
profile.profile_time_utc <= T
```

При нарушении cutoff-контракта кодирование блокируется.

Токенайзер не должен молча отбрасывать события после T, потому что это скрывает upstream leakage.

Downstream target не должен присутствовать во входной последовательности.

## 3. FEATURE SCHEMA

Feature Schema является общим контрактом preprocessing и токенайзера.

`event_type` хранится на верхнем уровне canonical event и не дублируется в `fields`.

Пример входного события:

```json
{
  "event_type": "TRANSFER",
  "fields": {
    "amount_base_bucket": "bucket_17",
    "currency": "KZT",
    "channel": "MOBILE"
  }
}
```

Для каждого поля схема задаёт:

- имя;
- стабильный порядок;
- тип подготовленного значения;
- обязательность (required) — нужна токенайзеру для правил §17;
- `vocabulary_policy`;
- closed-set domain, если он существует;
- multivalue policy;
- приоритет при `max_tokens_per_event`;
- model inclusion flag.

Базовые `vocabulary_policy`:

1. `closed_set` — фиксированный набор значений; `min_count` не применяется.
2. `bucket_closed_set` — все configured bucket values и MISSING включаются всегда.
3. `frequency_pruned` — open-set значения проходят `min_count` и могут стать RARE.
4. `excluded` — поле не токенизируется.

Пример:

```yaml
TRANSFER:
  fields:
    - name: amount_base_bucket
      vocabulary_policy: bucket_closed_set
      domain: [bucket_0, bucket_1, ..., bucket_63, MISSING]
      priority: 1
    - name: currency
      vocabulary_policy: closed_set
      domain: [KZT, USD, EUR, MISSING]
      priority: 2
    - name: merchant_category
      vocabulary_policy: frequency_pruned
      priority: 3
    - name: channel
      vocabulary_policy: closed_set
      domain: [MOBILE, WEB, BRANCH, MISSING]
      priority: 4
```

Токенайзер всегда эмитит `event_type` первым после `[EVT]`, затем читает `fields` в порядке Feature Schema.

Feature Schema покрывает и профильные поля (preprocessing §11.2): у каждого профильного поля есть своя `vocabulary_policy`.

## 4. MISSING, RARE И UNK

### 4.1. MISSING

`MISSING` уже должен быть создан preprocessing pipeline.

Пример:

```text
key:amount_base_bucket
value:amount_base_bucket:MISSING
```

Токенайзер не определяет причину пропуска и не заменяет сырые `None/NaN`.

### 4.2. RARE

`RARE` применяется только к полям с:

```text
vocabulary_policy = frequency_pruned
```

Формат:

```text
value:<field>:RARE
```

`RARE` означает, что значение встречалось в TRAIN, но не прошло `min_count`.

`RARE` запрещён для:

- `event_type`;
- bucket-полей;
- других `closed_set` полей;
- специальных токенов;
- key-токенов.

### 4.3. UNK

`[UNK]` применяется к новому open-set значению, отсутствующему после freeze Vocabulary.

Для `closed_set` или `bucket_closed_set` поля неизвестное значение означает несовместимость preprocessing/Feature Schema.

Пример:

```text
amount_base_bucket = bucket_99
```

если configured domain содержит только `bucket_0…bucket_63`, является schema violation и должно блокировать кодирование, а не превращаться в RARE или `[UNK]`.

**Различие:**

- `MISSING` — preprocessing знает, что значения нет;
- `RARE` — TRAIN-значение open-set поля было редким;
- `[UNK]` — новое open-set значение отсутствует в frozen Vocabulary;
- schema violation — значение нарушает closed-set domain.

## 5. СПЕЦИАЛЬНЫЕ ТОКЕНЫ

**Минимальный набор:**

```text
[PAD]
[UNK]
[MASK]
[EVT]
[USR]
```

Квадратные скобки — соглашение об именовании.
Это не техническое требование языка программирования.

### 5.1. [EVT]

**Назначение:**

- обозначить начало события;
- создать выделенную позицию для представления события.

[EVT] добавляется первым токеном каждого события.

### 5.2. [USR]

**Назначение:**

- обозначить начало профиля;
- создать выделенную позицию для представления профиля клиента.

[USR] добавляется первым токеном профиля.

### 5.3. [PAD]

**Назначение:**

- выровнять последовательности в batch.

[PAD] заранее имеет зарезервированный token_id.

**Правило текущего регламента:**

используется только right-padding.

**Пример:**

```text
[3, 7, 13]
→ [3, 7, 13, 0, 0]
```

Запрещено смешивать right-padding и left-padding между TRAIN и Production.

### 5.4. [MASK]

Используется только в обучающих задачах masking.

### 5.5. [UNK]

Используется только для токенов, отсутствующих в замороженном Vocabulary.

## 6. ФОРМИРОВАНИЕ ПРОФИЛЯ [USR]

**Пример профиля:**

```text
{
    "region": "ALMATY",
    "employment": "EMPLOYED",
    "account_age_bucket": "bucket_12"
}
```

**Токены:**

```text
[USR]
key:region
value:region:ALMATY
key:employment
value:employment:EMPLOYED
key:account_age_bucket
value:account_age_bucket:bucket_12
```

Профиль должен быть рассчитан на cutoff T.

В профиль необходимо включать life-long признаки, чтобы truncation событий
не удалял долгосрочный контекст клиента.

**Минимально рекомендуемые life-long признаки:**

- account_age_bucket;
- first_seen_age_bucket;
- first_topup_age_bucket;
- lifetime_event_count_bucket;
- lifetime_transaction_count_bucket;
- lifetime_product_count_bucket.

Все life-long признаки рассчитываются только по данным timestamp <= T.

## 7. ФОРМИРОВАНИЕ СОБЫТИЯ [EVT]

Canonical event содержит `event_type` на верхнем уровне и подготовленные признаки в `fields`.

Пример:

```json
{
  "event_type": "TRANSFER",
  "fields": {
    "amount_base_bucket": "bucket_17",
    "currency": "KZT"
  }
}
```

Токенайзер формирует:

```text
[EVT]
key:event_type
value:event_type:TRANSFER
key:amount_base_bucket
value:amount_base_bucket:bucket_17
key:currency
value:currency:KZT
```

Правила:

1. `[EVT]` добавляется первым.
2. `event_type` берётся из top-level поля.
3. `event_type` не должен дублироваться в `fields`.
4. После event_type поля читаются в порядке Feature Schema.
5. Value-токен использует формат `value:<field>:<value>`.

## 8. ПОДГОТОВЛЕННЫЕ BUCKET-ПОЛЯ

Токенайзер не работает с исходными непрерывными числами.

Он получает уже подготовленное категориальное значение:

```text
amount_base_bucket = bucket_17
```

и токенизирует его как обычное closed-set поле:

```text
key:amount_base_bucket
value:amount_base_bucket:bucket_17
```

Для токенайзера `bucket_17` — категориальная метка, а не число.

Токенайзер не выполняет:

- numeric parsing;
- NaN/Inf handling;
- FX conversion;
- bucketization;
- clipping;
- fit или transform bucket_edges.

Исключение — `time_delta_edges` time-канала (§10.2): они применяются к пересчитанной дельте после truncation и не относятся к numeric bucketization полей события.

Все configured значения bucket-поля включаются в Vocabulary независимо от частоты:

```text
bucket_0 … bucket_63
MISSING
```

`min_count`, `RARE` и `[UNK]` к корректным bucket values не применяются.

### 8.1. Роль bucket metadata

`bucket_edges` могут поставляться в model package как read-only metadata только для:

- decode;
- объяснения интервала;
- model cards;
- observability;
- проверки совместимости артефактов.

Пример decode:

```text
value:amount_base_bucket:bucket_17
→ интервал 10 000–15 000 KZT
```

Bucket metadata не участвуют в преобразовании подготовленного bucket value в `token_id`.

Изменение bucket metadata требует совместимой preprocessing/model версии, но токенайзер не применяет edges повторно.

## 9. ГРАНИЦА С FX И BUCKETIZATION

FX-нормализация, numeric validation, bucketization и clipping полностью принадлежат preprocessing pipeline.

Токенайзер ожидает:

```text
amount_base_bucket = bucket_N | MISSING
currency = canonical currency code | MISSING
```

Он не использует исторические курсы и не выбирает bucket по сумме.

Если на вход пришли raw `amount`, `NaN`, строковая сумма или неготовое FX-значение, кодирование блокируется как нарушение preprocessing contract.

## 10. КОДИРОВАНИЕ ВРЕМЕНИ

Время передаётся отдельным time-embedding channel, а не key/value-токенами.

Preprocessing передаёт (форма canonical event — по preprocessing §5/§32.2):

- `timestamp_utc`;
- `calendar_timezone`;
- `calendar_time_features.hour_of_day_local`;
- `calendar_time_features.day_of_week_local`;
- при необходимости другие локальные календарные признаки.

Токенайзер/model-input pipeline не вычисляет локальный час из UTC заново.

### 10.1. delta_from_previous_event после truncation

Финальная `delta_from_previous_event` рассчитывается только после выбора model window.

Порядок:

1. собрать полную последовательность событий;
2. применить truncation по целым событиям;
3. взять только surviving events;
4. пересчитать delta между соседними surviving events;
5. для первого события окна использовать специальное значение `WINDOW_START`.

`FIRST_EVENT` используется только если событие действительно является первым известным событием lifetime и история не была обрезана перед ним.

Запрещено оставлять у первого surviving event дельту до события, которое было удалено truncation.

Различие `FIRST_EVENT`/`WINDOW_START` реализуемо, только если токенайзер получает полный prepared ≤T timeline (окно с индексом 0 полного timeline → `FIRST_EVENT`, иначе `WINDOW_START`). Если в production грузится только недавний срез истории, preprocessing обязан передавать флаг `lifetime_first` на самом раннем известном событии клиента; без гарантии «первое в lifetime» токенайзер назначает `WINDOW_START`.

### 10.2. Минимальный time channel

Для каждого surviving event:

- `delta_from_previous_event` или `WINDOW_START`;
- `hour_of_day_local`;
- `day_of_week_local`.

После пересчёта дельта преобразуется через versioned `time_delta_edges`
или через утверждённую continuous projection.

`WINDOW_START` и `FIRST_EVENT` — два зарезервированных значения delta-канала: для bucket-варианта это два выделенных reserved bucket id вне обычных `time_delta_edges`; для continuous-варианта — два обучаемых sentinel-эмбеддинга. Они не являются vocab-спецтокенами (§5) и не расходуют `max_tokens_per_event`.

`time_delta_edges` относятся к time-input component, а не к numeric bucketization полей события.

Time vector связывается с позицией `[EVT]` и не расходует `max_tokens_per_event`.

### 10.3. Календарный часовой пояс

Календарные признаки должны приходить из preprocessing в бизнес-локальной IANA timezone.

UTC используется для instant и сортировки, но не как обязательная зона для поведенческих hour/day features.

Tokenizer проверяет наличие timezone metadata и совместимые `calendar_timezone_policy_version` и (при bucket-варианте дельты) `time_delta_edges_version`.

## 11. МНОГОЗНАЧНЫЕ ПОЛЯ

**Пример:**

```text
products = ["CARD", "LOAN", "DEPOSIT"]
```

**Рекомендуемый формат:**

```text
key:products
value:products:CARD
value:products:LOAN
value:products:DEPOSIT
```

**Правила:**

- один key, несколько value;
- порядок значений сохраняется как передан preprocessing (значим — хронология; незначим — уже отсортирован);
- ограничение количества значений (`max_values_per_field`, priority, `count_bucket`) выполняет preprocessing (§21); токенайзер эмитит уже ограниченный набор и не отбрасывает значения сам;
- дополнительное сокращение — только в рамках per-event token budget (§17), не по `max_values_per_field`.

## 12. ВЫСОКОКАРДИНАЛЬНЫЕ ПРИЗНАКИ

Подготовка высококардинальных полей (исключение технических ID, замена бизнес-сущностей категориями, whitelist частых значений, hashing) выполняется preprocessing (§22). На вход токенайзеру приходит уже нормализованное значение поля.

Ответственность токенайзера ограничена open-set полями с `vocabulary_policy = frequency_pruned`:

- редкое TRAIN-значение (`count < min_count`) → field-scoped `RARE` (§14.2);
- новое значение после freeze → `[UNK]`.

Токенайзер не исключает ID, не строит категории и не хеширует.

## 13. ТЕКСТОВЫЕ ПОЛЯ

Удаление свободного текста и применение text policy выполняет preprocessing (§23); Feature Schema не содержит текстовых полей. Токенайзер лишь валидирует, что запрещённые поля не пришли.

**Правило текущего регламента:**

свободный текст в базовой версии запрещён.

**Не допускаются напрямую:**

- payment_description;
- notification_text;
- support_message;
- raw merchant_name.

**Разрешены:**

- утверждённые категории;
- нормализованные справочные значения;
- заранее рассчитанные безопасные текстовые признаки.

BPE/subword-токенизация может быть добавлена только в отдельной версии архитектуры.

До этого момента merchant_name_subwords и аналогичные поля запрещены
в Feature Schema.

## 14. MIN_COUNT И ПОСТРОЕНИЕ VOCABULARY

Vocabulary строится только на подготовленных TRAIN-данных.

Обязательный параметр для open-set полей:

```text
min_count = 20
```

`min_count` применяется только к:

```text
vocabulary_policy = frequency_pruned
```

### 14.1. Токены, освобождённые от min_count

Всегда включаются независимо от частоты:

- special tokens;
- key-токены Feature Schema;
- `event_type` domain;
- все значения `closed_set`;
- все configured bucket values;
- `MISSING` для всех полей Feature Schema — событийных и профильных, закрытых и open-set (`value:<field>:MISSING` не проходит `min_count` и никогда не становится RARE);
- whitelist business-critical values.

Следовательно, редкий хвостовой бакет:

```text
value:amount_base_bucket:bucket_63
```

не превращается в RARE, даже если встретился меньше `min_count`.

### 14.2. Open-set значения

Для `frequency_pruned` поля:

```text
count >= min_count
→ отдельный token

count < min_count
→ value:<field>:RARE
```

Новое production-значение такого поля после freeze:

```text
→ [UNK]
```

### 14.3. Алгоритм BUILD

1. загрузить prepared TRAIN и проверить preprocessing hash;
2. добавить special tokens;
3. добавить key-токены;
4. добавить все closed-set domains;
5. посчитать частоты только для frequency-pruned values;
6. применить min_count;
7. создать field-scoped RARE;
8. назначить token_id в каноническом порядке;
9. сохранить и заморозить Vocabulary.

Целевой размер Vocabulary является конфигурацией модели, но не может достигаться удалением легитимных closed-set bucket values.

Стартовый ориентир размера: 50 000–250 000 токенов. Это не универсальная норма; финальный диапазон зависит от модели, памяти и числа доменов.

### 14.4. Детерминизм BUILD PHASE при параллельной обработке

Одинаковые prepared TRAIN-данные и одинаковая конфигурация должны давать байт-в-байт одинаковые tokenizer artifacts независимо от числа workers.

Одинаковыми должны быть:

- open-set token counts;
- retained tokens;
- RARE mappings;
- closed-set domains;
- token_id;
- serialized Vocabulary;
- tokenizer content hash.

#### 14.4.1. Детерминированный подсчёт частот

1. Prepared input files сортируются по stable file ID или canonical path.
2. Партиции обрабатываются в стабильном порядке.
3. Локальные open-set counters объединяются в строгом порядке.
4. token_id не назначается в порядке прихода worker results.
5. Closed-set domains загружаются из versioned preprocessing artifacts.
6. Перед назначением ID токены сортируются согласно каноническому порядку (§14.4.1.1).
7. Vocabulary сериализуется с фиксированным порядком ключей и по правилам сериализации preprocessing §29.1.

Tokenizer BUILD не выполняет reservoir sampling и не рассчитывает bucket_edges.
Эти операции принадлежат preprocessing BUILD.

##### 14.4.1.1. Канонический порядок token_id

Порядок фиксирован и не зависит от реализации:

1. Зарезервированный блок специальных токенов с фиксированными id: `[PAD]=0`, `[UNK]=1`, `[MASK]=2`, `[EVT]=3`, `[USR]=4`. `right-padding` (§19) требует `[PAD]=0` — это не конфигурируется.
2. Затем следующие блоки в строгом порядке; токены внутри каждого блока сортируются по возрастанию UTF-8 byte order канонической строки токена:
   - key-токены (`key:<field>`);
   - все closed-set domains (`event_type`, `bucket_closed_set`, `closed_set`) в форме `value:<field>:<value>`;
   - retained frequency_pruned значения (`value:<field>:<value>`), включая всегда включаемый `value:<field>:MISSING` каждого open-set поля;
   - field-scoped RARE (`value:<field>:RARE`).
3. token_id назначается последовательно при обходе блоков.

Глобальная сортировка всех токенов по UTF-8 без блоков запрещена: она сместила бы фиксированные id специальных токенов.

#### 14.4.2. Проверка детерминизма

Перед релизом Vocabulary собирается минимум в двух конфигурациях:

- single-worker;
- multi-worker.

Tokenizer artifacts сравниваются байт-в-байт.

Несовпадение является блокирующей ошибкой.

#### 14.4.3. Golden-vector conformance set

Проверка §14.4.2 покрывает лишь single vs multi-worker одной реализации. Для идентичности **между реализациями** обязателен замороженный golden-vector набор токенайзера.

Состав:

1. `golden_prepared` — фиксированный prepared-вход (профиль + события); это ровно `golden_expected` из preprocessing §29.2, поэтому два набора складываются в сквозную проверку raw → token_id.
2. `golden_tokens` — заморожённые ожидаемые `token_id` каждого события и профиля, значения time-канала surviving-окна (delta buckets/sentinels, включая `WINDOW_START`/`FIRST_EVENT`) плюс `tokenizer_state_sha256`.

Правила:

1. Значения заполняются первым эталонным BUILD/ENCODE на замороженном Vocabulary; вручную не составляются.
2. Любая реализация обязана из `golden_prepared` получить `golden_tokens` байт-в-байт — включая порядок полей, `RARE`/`[UNK]`/`MISSING` и `WINDOW_START`.
3. Несовпадение — блокирующая ошибка релиза.
4. Golden set версионируется с `vocabulary_version`/`tokenizer_version` и прогоняется в CI.
5. Отдельный негативный кейс: prepared событие со значением вне closed-set domain (например `bucket_99`) — ожидаемый и замороженный результат: блокировка encode (closed-set violation), а не token_id. Позитивная цепочка golden-проверку блокировок дать не может, поэтому кейс отдельный.

## 15. КОДИРОВАНИЕ В TOKEN_ID

Id специальных токенов (`[PAD]=0 … [USR]=4`) — нормативны и фиксированы (§14.4.1.1); id остальных токенов иллюстративны и определяются каноническим порядком.

**Пример Vocabulary:**

```text
[PAD] → 0
[UNK] → 1
[MASK] → 2
[EVT] → 3
[USR] → 4
key:event_type → 5
value:event_type:TRANSFER → 6
```

**Было:**

```text
[EVT]
key:event_type
value:event_type:TRANSFER
```

**Стало:**

```text
[3, 5, 6]
```

**Один общий Vocabulary используется для:**

- профиля;
- всех типов событий;
- TRAIN;
- Validation;
- Test;
- Production.

## 16. СБОРКА ПОСЛЕДОВАТЕЛЬНОСТИ КЛИЕНТА

Базовый формат token sequence:

```text
[USR]
profile fields...

[EVT]
event_type + event 1 fields...

[EVT]
event_type + event 2 fields...
```

События уже приходят в детерминированном хронологическом порядке от preprocessing.

Parallel event metadata (форма canonical event — по preprocessing §5/§32.2, единый источник истины):

- `event_id`;
- `ordering_key`;
- `timestamp_utc`;
- `calendar_timezone`;
- `calendar_time_features.hour_of_day_local`;
- `calendar_time_features.day_of_week_local`;
- `lifetime_first` (опционально, §10.1).

После выбора окна SequenceBuilder пересчитывает window-relative `delta_from_previous_event`.

Границы определяются `[USR]` и `[EVT]`.

## 17. MAX_TOKENS_PER_EVENT

**Обязательный лимит:**

max_tokens_per_event = 24

**Лимит включает:**

- [EVT];
- key-токены;
- value-токены.

Time embedding в этот лимит не входит.

Feature Schema должна задавать priority полей.

**Пример для TRANSFER:**

```text
1. event_type
2. amount_base_bucket
3. currency
4. direction
5. channel
6. mcc
7. merchant_category
```

**При превышении лимита:**

1. сохраняется [EVT];
2. сохраняются обязательные поля;
3. сохраняются поля с высоким priority;
4. сначала режутся многозначные поля;
5. затем удаляются поля с низким priority;
6. нельзя оставлять key без value;
7. нельзя разрезать значение на некорректный фрагмент.

Если после сохранения `[EVT]` и всех обязательных полей лимит всё ещё превышен — это ошибка Feature Schema/preprocessing (обязательные поля не помещаются в `max_tokens_per_event`), а не повод резать обязательное поле: кодирование блокируется как нарушение контракта.

## 18. TRUNCATION ПОЛНОЙ ИСТОРИИ

Полная подготовленная история хранится upstream.

Truncation выполняется в model-input pipeline по token budget.

Обязательный параметр: `max_sequence_length` (пример: 8000 token_id). Фиксируется в sequence config и входит в `tokenizer_state_sha256`.

Правила:

1. профиль `[USR]` сохраняется целиком;
2. события добавляются целиком;
3. событие не разрезается посередине;
4. в production сохраняются самые свежие события;
5. truncation выполняется по границам `[EVT]`;
6. после выбора surviving events пересчитывается `delta_from_previous_event`;
7. первое событие окна получает `WINDOW_START`;
8. дельта до отброшенного события не передаётся модели.

Для TRAIN допустим random window при фиксированной и versioned window policy.

## 19. RIGHT-PADDING И ATTENTION_MASK

**Правило текущего регламента:**

используется только right-padding.

**Пример:**

```text
token_ids:
[3, 5, 6]
```

**padded:**

```text
[3, 5, 6, 0, 0]
```

**attention_mask:**

```text
[1, 1, 1, 0, 0]
```

**Где:**

1 — учитывать;
0 — игнорировать.

**attention_mask:**

```text
- создаётся автоматически при batch assembly;
- не входит в Vocabulary;
- должна полностью совпадать с позициями [PAD].
```

Для снижения PAD ratio применяется batching by length.

## 20. MASKING 80/10/10

**Базовое правило:**

маскируются только value-токены.

**Не маскируются:**

- key-токены;
- value-токен `event_type`;
- [EVT];
- [USR];
- [PAD];
- [UNK];
- [MASK].

**Причина:**

- порядок key-токенов детерминирован Feature Schema;
- предсказание key-токенов создаёт слишком простую задачу;
- value `event_type` исключён по той же причине: набор незамаскированных key-токенов события определяется Feature Schema конкретного event_type и выдаёт его почти однозначно — masked-предсказание тривиально;
- loss должен расходоваться на восстановление содержательных значений.

**Базовая стратегия masking:**

**для каждой выбранной value-позиции:**

- 80% — заменить на [MASK];
- 10% — заменить на случайный допустимый токен того же поля;
- 10% — оставить исходный токен.

Во всех трёх случаях label содержит исходный token_id.

Случайная замена (10%) выбирается равномерно из value-токенов того же поля, тем же seeded-генератором (§21); `RARE` допустим как замена для frequency_pruned поля, а `MISSING`, `[UNK]` и служебные токены — нет. Value-токены `MISSING`/`RARE` могут выбираться как masked-позиции и предсказываются наравне с обычными значениями.

**Пример:**

```text
исходно:
value:currency:KZT
```

**80%:**

[MASK]

**10%:**

value:currency:USD

**10%:**

value:currency:KZT

**Label во всех случаях:**

```text
value:currency:KZT
```

**Запрещено:**

- подставлять [PAD];
- подставлять token другого несовместимого поля;
- считать loss на немаскированных позициях;
- использовать masking в production inference.

**Базовая masking probability:**

```text
mask_probability = 0.15
```

Стратегия и вероятность являются гиперпараметрами.

## 21. ДЕТЕРМИНИЗМ MASKING

**Для воспроизводимости экспериментов обязательно сохраняются:**

- random seed;
- mask_probability;
- 80/10/10 ratios;
- версии библиотек;
- порядок выборки;
- distributed worker seed policy;
- random window policy.

**Пример:**

```text
masking_config = {
    "seed": 42,
    "mask_probability": 0.15,
    "replace_with_mask": 0.80,
    "replace_with_random": 0.10,
    "keep_original": 0.10
}
```

TRAIN может оставаться стохастическим,
но повторный эксперимент с тем же config должен быть воспроизводим.

## 22. BATCH ASSEMBLY

**Для каждого batch:**

1. взять последовательности похожей длины;
2. применить right-padding;
3. построить attention_mask;
4. связать event time metadata с позициями [EVT];
5. при TRAIN применить masking 80/10/10;
6. сформировать labels;
7. передать token_ids, attention_mask, time_features и labels в модель.

## 23. ОБРАТНОЕ ДЕКОДИРОВАНИЕ

**Необходимо поддерживать:**

token_id → строковый токен.

**inverse_vocabulary используется для:**

- проверки токенизации;
- анализа production input;
- поиска несовместимых версий;
- отладки RARE/UNK;
- проверки truncation.

## 24. ВЕРСИОНИРОВАНИЕ

Единый совместимый комплект. Upstream-версии используют единый словарь имён из preprocessing §30 — токенайзер не вводит собственных имён для upstream-артефактов:

- preprocessing_version;
- preprocessing_state_sha256;
- feature_schema_version;
- closed_set_domains_version;
- bucket_field_domains_version;
- bucket_edges_version (только как decode-metadata, см. §8.1);
- calendar_timezone_policy_version;
- time_delta_edges_version (при bucket-варианте дельты);
- vocabulary_version;
- tokenizer_version;
- masking_version;
- sequence_builder_version;
- model_version.

Tokenizer package не владеет FX rules, numeric parsing rules или bucket transform logic.

Он только проверяет совместимость upstream preprocessing state.

### 24.1. Content hash состояния токенайзера

`tokenizer_state_sha256` рассчитывается по канонически сериализованному состоянию:

- Vocabulary;
- inverse Vocabulary;
- RARE mappings;
- Feature Schema projection для токенизации;
- closed-set domains;
- special tokens;
- tokenizer config;
- masking config;
- sequence config;
- optional bucket metadata для decode;
- BPE artifacts, если появятся в будущей версии.

Каноническая сериализация чисел и артефактов — по правилам preprocessing §29.1 (единый формат float, запрет `NaN/Inf`, `-0.0 → 0.0`), чтобы `tokenizer_state_sha256` был байт-идентичен между реализациями.

При загрузке проверяются оба hash:

```text
preprocessing_state_sha256
tokenizer_state_sha256
```

Несовпадение любого hash блокирует inference.

## 25. PRODUCTION-МОНИТОРИНГ

Токенайзер считает только метрики своей зоны ответственности.

### 25.1. UNK rate

Только для open-set `frequency_pruned` полей:

- <= 1% — normal;
- > 1% — warning;
- > 5% — critical.

### 25.2. RARE rate

Контролируется по каждому frequency-pruned полю относительно TRAIN baseline.

### 25.3. Closed-set violation rate

Любое неизвестное значение для:

- `event_type`;
- bucket-поля;
- closed-set поля

является нарушением контракта.

Порог:

```text
> 0 → critical
```

### 25.4. Truncation rate

- > 20% клиентов — warning;
- > 50% — пересмотр window strategy.

### 25.5. PAD ratio

- > 30% — warning;
- > 60% — требуется batching by length.

### 25.6. WINDOW_START rate

Контролируется доля окон, где history была обрезана и первый event получил `WINDOW_START`.

FX missing, numeric parsing, clipping, source schema и MISSING upstream metrics принадлежат preprocessing monitoring и не пересчитываются токенайзером.

## 26. ОБЯЗАТЕЛЬНЫЕ АРТЕФАКТЫ

Необходимо сохранять:

- `vocabulary.json`;
- `inverse_vocabulary.json`;
- `rare_mapping.json`;
- `token_counts_open_set.json`;
- `tokenizer_feature_schema.json`;
- `closed_set_domains.json`;
- `special_tokens.json`;
- `tokenizer_config.json`;
- `masking_config.json`;
- `sequence_config.json`;
- `time_input_contract.json`;
- optional `bucket_metadata.json` только для decode;
- `preprocessing_version`;
- `preprocessing_state_sha256`;
- `tokenizer_state_sha256`;
- deterministic build config;
- golden_vectors (§14.4.3);
- code commit/hash;
- TRAIN dataset identifier;
- build timestamp.

`time_input_contract.json` описывает контракт time-канала: ссылки на upstream `calendar_timezone_policy_version` и (при bucket-варианте) `time_delta_edges_version`, представление `WINDOW_START`/`FIRST_EVENT` (§10.2) и связь time-vector с позицией `[EVT]`.

## 27. ПРОВЕРКИ КАЧЕСТВА

### A. INPUT CONTRACT

1. `preprocessing_state_sha256` совпадает с ожидаемым.
2. События уже имеют `timestamp <= T`.
3. `event_type` находится top-level и не дублируется в fields.
4. Fields соответствуют tokenizer projection Feature Schema.
5. Raw numeric, FX и необработанные NaN/None отсутствуют.
6. Bucket fields содержат только configured domain.
7. Локальные hour/day и timezone metadata присутствуют.

### B. VOCABULARY

8. Vocabulary построен только на prepared TRAIN.
9. `min_count` применяется только к frequency-pruned полям.
10. Bucket values освобождены от min_count.
11. Closed-set values освобождены от min_count.
12. Редкий `bucket_63` сохраняет собственный token.
13. RARE создаётся только для open-set TRAIN values.
14. Новые open-set values переходят в `[UNK]`.
15. Closed-set unknown value блокирует кодирование.
16. Key-токены и special tokens включены всегда.

### C. EVENT TOKENIZATION

17. `[EVT]` стоит первым.
18. `event_type` эмитится первым полем после `[EVT]`.
19. Остальные поля идут по Feature Schema.
20. Value format соответствует `value:<field>:<value>`.
21. Key не остаётся без value после event limiting.
22. max_tokens_per_event соблюдается.
23. Time metadata не расходует token budget.

### D. SEQUENCE И TIME

24. Timeline order совпадает с preprocessing.
25. Truncation выполняется по целым `[EVT]` блокам.
26. После truncation delta_prev пересчитывается.
27. Первое surviving event получает `WINDOW_START`.
28. Дельта до отброшенного event отсутствует.
29. Истинный `FIRST_EVENT` не смешивается с `WINDOW_START`.
30. Hour/day используются из local-time preprocessing metadata.
31. Time vector связан с правильной позицией `[EVT]`.

### E. PADDING И MASKING

32. Используется только right-padding.
33. attention_mask совпадает с `[PAD]`.
34. Masking применяется только к value-токенам.
35. `[EVT]`, `[USR]`, key, value `event_type` и служебные токены не маскируются.
36. 80/10/10 соблюдается.
37. Random replacement выбирается из допустимого поля.
38. Seed и masking config сохранены.

### F. BUILD И INTEGRITY

39. Single-worker и multi-worker Vocabulary совпадают байт-в-байт.
40. token_id не зависит от worker completion order.
41. Canonical serialization фиксирована.
42. `tokenizer_state_sha256` рассчитан.
43. Оба preprocessing/tokenizer hash проверяются при загрузке.
44. Bucket metadata не используется для encode.
45. Decode bucket interval соответствует compatible preprocessing metadata.

### G. PRODUCTION

46. UNK rate считается только для open-set fields.
47. Closed-set violation rate равен нулю.
48. RARE rate контролируется.
49. Truncation и WINDOW_START rate контролируются.
50. PAD ratio контролируется.
51. Tokenizer не пересчитывает FX, numeric parsing или clipping metrics.

### H. GOLDEN-VECTORS

52. `golden_prepared`/`golden_tokens` существуют и версионированы.
53. Эталонная реализация воспроизводит `golden_tokens` из `golden_prepared` байт-в-байт.
54. Несовпадение golden-vector блокирует релиз.

## 28. РЕКОМЕНДУЕМАЯ СТРУКТУРА КОДА

1. `PreparedInputValidator`
   - preprocessing hash;
   - top-level event_type;
   - Feature Schema projection;
   - closed-set domains;
   - отсутствие raw numeric/FX.

2. `VocabularyBuilder`
   - special tokens;
   - key tokens;
   - closed-set domains;
   - open-set token counts;
   - min_count;
   - RARE;
   - deterministic ordering.

3. `Tokenizer`
   - `[USR]`;
   - `[EVT]`;
   - top-level event_type emission;
   - key/value;
   - encode/decode;
   - `[UNK]`.

4. `EventLimiter`
   - max_tokens_per_event;
   - field priority;
   - key/value integrity.

5. `SequenceBuilder`
   - profile and events;
   - truncation;
   - surviving event metadata;
   - `WINDOW_START`;
   - delta_prev recomputation.

6. `TimeFeatureAdapter`
   - local hour/day input;
   - window-relative delta;
   - event-to-`[EVT]` alignment.

7. `BatchCollator`
   - right-padding;
   - attention_mask;
   - batching by length.

8. `MaskingCollator`
   - value-only masking;
   - 80/10/10;
   - labels;
   - seed.

9. `ArtifactHasher`
   - canonical serialization;
   - SHA-256;
   - compatibility checks.

10. `TokenizerMonitor`
    - UNK;
    - RARE;
    - closed-set violations;
    - truncation;
    - WINDOW_START;
    - PAD ratio.

## 29. КАНОНИЧЕСКИЙ ПОРЯДОК РЕАЛИЗАЦИИ

### 29.1. BUILD PHASE — только prepared TRAIN

1. Загрузить prepared TRAIN.
2. Проверить `preprocessing_state_sha256`.
3. Загрузить tokenizer Feature Schema projection.
4. Добавить special tokens.
5. Добавить key-токены.
6. Добавить `event_type` domain.
7. Добавить все closed-set domains.
8. Добавить все bucket domains и MISSING независимо от частоты.
9. Посчитать частоты только open-set frequency-pruned values.
10. Применить min_count только к open-set values.
11. Сформировать field-scoped RARE.
12. Назначить token_id в каноническом порядке.
13. Проверить single/multi-worker byte equality.
14. Сохранить optional bucket metadata только для decode.
15. Рассчитать `tokenizer_state_sha256`.
16. Заморозить Vocabulary.

### 29.2. ENCODE PHASE — каждый prepared пример

1. Проверить preprocessing/tokenizer hash.
2. Валидировать top-level `event_type`.
3. Валидировать fields и closed-set domains.
4. Сформировать `[USR]` profile block.
5. Для события добавить `[EVT]`.
6. Эмитить top-level event_type первым.
7. Эмитить остальные поля по Feature Schema.
8. Применить RARE mapping только к open-set fields.
9. Новые open-set values преобразовать в `[UNK]`.
10. Closed-set violation блокирует encode.
11. Преобразовать токены в token_id.
12. Собрать полную последовательность.
13. Применить truncation по целым событиям.
14. Пересчитать delta_prev на surviving window.
15. Первому surviving event назначить `WINDOW_START`.
16. Связать local hour/day и time vector с `[EVT]`.
17. Собрать batch.
18. Применить right-padding.
19. Построить attention_mask.
20. В TRAIN применить value-only masking 80/10/10.
21. Сформировать labels.
22. Передать входы в embedding и Transformer.

## 30. ИТОГОВАЯ СХЕМА

**BUILD:**

```text
prepared TRAIN
→ preprocessing hash validation
→ special/key tokens
→ event_type domain
→ closed-set + bucket domains
→ open-set token counts
→ min_count only for open-set
→ field-scoped RARE
→ Vocabulary
→ deterministic byte comparison
→ tokenizer_state_sha256
→ freeze
```

**ENCODE:**

```text
prepared profile/events
→ validate top-level event_type and fields
→ [USR] / [EVT]
→ event_type token first
→ key/value tokens
→ RARE only for open-set
→ [UNK] only for new open-set values
→ token_id
→ sequence
→ truncation by whole events
→ recompute delta_prev
→ WINDOW_START for first surviving event
→ local calendar time channel
→ right-padding
→ attention_mask
→ value-only masking during TRAIN
→ embedding
→ Transformer
```

Bucket edges, FX rates, numeric parsing and clipping do not participate in tokenizer ENCODE.

## 31. КЛЮЧЕВЫЕ ПРАВИЛА

1. Токенайзер работает только с prepared data.
2. Numeric parsing, FX, bucketization и clipping принадлежат preprocessing.
3. Bucket value является обычной closed-set категорией.
4. Числовые bucket edges полей события используются токенайзером только как optional decode metadata; `time_delta_edges` time-канала применяются к дельте после truncation (§10.2).
5. `event_type` хранится top-level и эмитится токенайзером первым.
6. `event_type` не дублируется в fields.
7. `min_count` применяется только к open-set frequency-pruned полям.
8. Bucket values никогда не схлопываются в RARE.
9. Closed-set values никогда не схлопываются в RARE.
10. Новое closed-set значение является contract violation.
11. `[UNK]` предназначен только для нового open-set значения.
12. Truncation выполняется по целым событиям.
13. delta_prev пересчитывается после truncation.
14. Первое surviving event получает `WINDOW_START`.
15. Hour/day берутся из локальных календарных признаков preprocessing.
16. UTC используется для instant и ordering, а не как поведенческий local hour.
17. Time metadata не расходует max_tokens_per_event.
18. Используется только right-padding.
19. Masking применяется только к value-токенам.
20. Preprocessing и tokenizer hashes проверяются при загрузке.
21. Tokenizer не пересчитывает preprocessing monitoring metrics.
22. TRAIN и Production используют один frozen tokenizer state.

КОНЕЦ ДОКУМЕНТА
