# Журнал изменений - Новые эвристики качества данных

## Дата: 10 декабря 2025

### Добавлены эвристики качества данных

1. **`has_constant_columns`** - выявление колонок с константными значениями (где все значения одинаковые)
2. **`has_suspicious_id_duplicates`** - проверка уникальности идентификаторов (user_id, id, uuid и т.п.)

### Добавлены новые параметры команды `report`

1. **`--title`** - настраиваемый заголовок отчета
2. **`--top-k-categories`** - количество top-значений для категориальных признаков

---

## Измененные файлы

### 1. `src/eda_cli/core.py`

**Функция:** `compute_quality_flags()`

**Добавлено:**
- Логика определения константных колонок
- Новые флаги в результате функции
- Влияние на общий скор качества

**Детали изменений:**
```python
# ДОБАВЛЕНО: Новая эвристика качества - константные колонки
constant_columns = []
for col in summary.columns:
    # Колонка считается константной, если у неё только 1 уникальное значение
    # и есть хотя бы одно не-null значение
    if col.unique == 1 and col.non_null > 0:
        constant_columns.append(col.name)

flags["has_constant_columns"] = len(constant_columns) > 0
flags["constant_columns"] = constant_columns
```

```python
# ДОБАВЛЕНО: Новая эвристика качества - подозрительные дубликаты ID
suspicious_id_columns = []
for col in summary.columns:
    # Ищем колонки с названиями, похожими на ID
    col_name_lower = col.name.lower()
    if any(id_pattern in col_name_lower for id_pattern in ['id', '_id', 'uuid', 'key']):
        # Проверяем, что количество уникальных значений меньше общего количества строк
        if col.unique < col.non_null and col.non_null > 0:
            suspicious_id_columns.append(col.name)

flags["has_suspicious_id_duplicates"] = len(suspicious_id_columns) > 0
flags["suspicious_id_columns"] = suspicious_id_columns
```

```python
# ДОБАВЛЕНО: Учет константных колонок в скоре качества
if flags["has_constant_columns"]:
    score -= 0.1  # константные колонки снижают качество
# ДОБАВЛЕНО: Учет подозрительных дубликатов ID в скоре качества
if flags["has_suspicious_id_duplicates"]:
    score -= 0.15  # дубликаты ID - серьезная проблема качества
```

---

### 2. `src/eda_cli/cli.py`

**Функция:** `report()` - секция генерации Markdown отчета

**Добавлено:**
- Вывод флага `has_constant_columns` в отчет
- Список константных колонок (если есть)
- Новые параметры командной строки
- Использование настраиваемого заголовка
- Информация о настройках отчета

**Детали изменений:**
```python
# ДОБАВЛЕНО: Вывод новой эвристики has_constant_columns
f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
if quality_flags['has_constant_columns']:
    constant_cols = ', '.join(quality_flags['constant_columns'])
    f.write(f"- Константные колонки: **{constant_cols}**\n")
# ДОБАВЛЕНО: Вывод новой эвристики has_suspicious_id_duplicates
f.write(f"- Подозрительные дубликаты ID: **{quality_flags['has_suspicious_id_duplicates']}**\n")
if quality_flags['has_suspicious_id_duplicates']:
    suspicious_cols = ', '.join(quality_flags['suspicious_id_columns'])
    f.write(f"- Колонки с дубликатами ID: **{suspicious_cols}**\n")
```

```python
# ДОБАВЛЕНО: Новые параметры командной строки
title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
top_k_categories: int = typer.Option(5, help="Количество top-значений для категориальных признаков."),
```

```python
# ДОБАВЛЕНО: Использование нового параметра top_k_categories
top_cats = top_categories(df, top_k=top_k_categories)
```

```python
# ДОБАВЛЕНО: Использование настраиваемого заголовка
f.write(f"# {title}\n\n")
```

```python
# ДОБАВЛЕНО: Информация о настройках отчета
f.write("\n### Настройки отчета\n\n")
f.write(f"- Максимум гистограмм: **{max_hist_columns}**\n")
f.write(f"- Top-K категорий: **{top_k_categories}**\n")
```

---

## Результат

### Новые поля в `compute_quality_flags()`:
- `has_constant_columns` (bool) - есть ли константные колонки
- `constant_columns` (list) - список имен константных колонок
- `has_suspicious_id_duplicates` (bool) - есть ли подозрительные дубликаты ID
- `suspicious_id_columns` (list) - список колонок с дубликатами ID

### Влияние на качество:
- Наличие константных колонок снижает общий скор качества на 0.1
- Наличие дубликатов ID снижает общий скор качества на 0.15

### Вывод в отчете:
- В разделе "Качество данных (эвристики)" появляются новые строки
- Показывается флаг наличия константных колонок
- При наличии - выводится список имен таких колонок
- Показывается флаг наличия подозрительных дубликатов ID
- При наличии - выводится список колонок с дубликатами ID

### Использование:

**Базовый отчет:**
```bash
eda-cli report data/example.csv --out-dir reports
```

**С новыми параметрами:**
```bash
eda-cli report data/example.csv \
  --title "Анализ данных клиентов" \
  --top-k-categories 8 \
  --out-dir reports
```

Результат будет в файле `reports/report.md`.

---

## Логика работы

### Константные колонки (`has_constant_columns`)
Колонка считается **константной**, если:
1. `col.unique == 1` (только одно уникальное значение)
2. `col.non_null > 0` (есть хотя бы одно не-null значение)

Такие колонки не несут информационной ценности для анализа данных.

### Подозрительные дубликаты ID (`has_suspicious_id_duplicates`)
Колонка считается **подозрительной на дубликаты ID**, если:
1. Название содержит паттерны: `'id'`, `'_id'`, `'uuid'`, `'key'`
2. `col.unique < col.non_null` (количество уникальных значений меньше не-null значений)
3. `col.non_null > 0` (есть хотя бы одно не-null значение)

Такие колонки указывают на проблемы с уникальностью идентификаторов.

---

## Новые файлы

### 3. `README.md`

**Создан новый файл** с полной документацией проекта:
- Описание всех команд CLI (`overview` и `report`)
- Документация новых параметров и их влияния на отчет
- Примеры использования
- Требования и структура проекта

**Примеры команд из README:**
```bash
# Отчет с настройками
eda-cli report data/sales.csv \
  --title "Анализ продаж за 2024 год" \
  --out-dir reports/sales_2024 \
  --max-hist-columns 10 \
  --top-k-categories 8
```

---

## Влияние новых параметров

### `--title`
- **Где используется:** Заголовок в `report.md`
- **По умолчанию:** "EDA-отчёт"
- **Пример:** `--title "Анализ клиентской базы"`

### `--top-k-categories`
- **Где используется:** Функция `top_categories()` и описание в отчете
- **По умолчанию:** 5
- **Влияние:** Определяет количество топ-значений в файлах `top_categories/*.csv`
- **Пример:** `--top-k-categories 10`

### Секция "Настройки отчета"
Новая секция в `report.md` показывает:
- Максимум гистограмм
- Top-K категорий

Это помогает понять, какие параметры использовались при генерации отчета.

---

## Добавлены тесты для новых эвристик

### 4. `tests/test_core.py`

**Добавлено 5 новых тестов** для проверки эвристик качества данных:

#### Тесты для `has_constant_columns`:
- **`test_has_constant_columns()`** - проверяет обнаружение константных колонок
- **`test_no_constant_columns()`** - проверяет случай отсутствия константных колонок

#### Тесты для `has_suspicious_id_duplicates`:
- **`test_has_suspicious_id_duplicates()`** - проверяет обнаружение дубликатов ID
- **`test_no_suspicious_id_duplicates()`** - проверяет случай отсутствия дубликатов ID

#### Общий тест:
- **`test_quality_score_with_new_heuristics()`** - проверяет влияние новых эвристик на скор качества

**Покрытие тестами:**
- ✅ Позитивные сценарии (проблемы обнаружены)
- ✅ Негативные сценарии (проблем нет)
- ✅ Корректность флагов (`True`/`False`)
- ✅ Корректность списков проблемных колонок
- ✅ Влияние на общий скор качества

**Тестовые данные:**
```python
# Константные колонки
df = pd.DataFrame({
    "constant_col": ["same", "same", "same", "same"],
    "another_constant": [42, 42, 42, 42],
    "normal_col": [10, 20, 30, 40]
})

# Дубликаты ID
df = pd.DataFrame({
    "user_id": [1, 2, 2, 3],  # дубликат
    "product_key": ["A", "A", "B", "B"],  # дубликаты
    "uuid": ["uuid1", "uuid2", "uuid1", "uuid3"]  # дубликаты
})
```

**Запуск тестов:**
```bash
pytest tests/test_core.py -v
```