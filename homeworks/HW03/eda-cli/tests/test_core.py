from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_has_constant_columns():
    """Тест эвристики has_constant_columns - проверка константных колонок."""
    # ДОБАВЛЕНО: Тест для эвристики константных колонок
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "constant_col": ["same", "same", "same", "same"],  # константная колонка
        "normal_col": [10, 20, 30, 40],
        "another_constant": [42, 42, 42, 42],  # еще одна константная
        "mixed_col": ["A", "B", "A", "C"]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_constant_columns установлен в True
    assert flags["has_constant_columns"] is True
    
    # Проверяем, что список константных колонок содержит правильные колонки
    constant_cols = flags["constant_columns"]
    assert "constant_col" in constant_cols
    assert "another_constant" in constant_cols
    assert len(constant_cols) == 2
    
    # Проверяем, что обычные колонки не попали в список константных
    assert "user_id" not in constant_cols
    assert "normal_col" not in constant_cols
    assert "mixed_col" not in constant_cols


def test_has_suspicious_id_duplicates():
    """Тест эвристики has_suspicious_id_duplicates - проверка дубликатов ID."""
    # ДОБАВЛЕНО: Тест для эвристики подозрительных дубликатов ID
    df = pd.DataFrame({
        "user_id": [1, 2, 2, 3],  # дубликат в ID колонке
        "session_id": [101, 102, 103, 104],  # уникальные ID
        "product_key": ["A", "A", "B", "B"],  # дубликаты в key колонке
        "name": ["Alice", "Bob", "Charlie", "David"],  # обычная колонка
        "uuid": ["uuid1", "uuid2", "uuid1", "uuid3"],  # дубликаты в uuid
        "regular_col": [10, 20, 30, 40]  # обычная колонка без ID паттернов
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_suspicious_id_duplicates установлен в True
    assert flags["has_suspicious_id_duplicates"] is True
    
    # Проверяем, что список подозрительных колонок содержит правильные колонки
    suspicious_cols = flags["suspicious_id_columns"]
    assert "user_id" in suspicious_cols  # есть дубликаты
    assert "product_key" in suspicious_cols  # есть дубликаты
    assert "uuid" in suspicious_cols  # есть дубликаты
    
    # Проверяем, что колонки без дубликатов или без ID паттернов не попали в список
    assert "session_id" not in suspicious_cols  # нет дубликатов
    assert "name" not in suspicious_cols  # не ID колонка
    assert "regular_col" not in suspicious_cols  # не ID колонка


def test_no_constant_columns():
    """Тест случая, когда константных колонок нет."""
    # ДОБАВЛЕНО: Тест отсутствия константных колонок
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4],
        "col2": ["A", "B", "C", "D"],
        "col3": [10.1, 20.2, 30.3, 40.4]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_constant_columns установлен в False
    assert flags["has_constant_columns"] is False
    
    # Проверяем, что список константных колонок пуст
    assert len(flags["constant_columns"]) == 0


def test_no_suspicious_id_duplicates():
    """Тест случая, когда подозрительных дубликатов ID нет."""
    # ДОБАВЛЕНО: Тест отсутствия подозрительных дубликатов ID
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],  # уникальные ID
        "session_id": [101, 102, 103, 104],  # уникальные ID
        "name": ["Alice", "Bob", "Alice", "David"],  # дубликаты в не-ID колонке
        "category": ["X", "Y", "X", "Z"]  # дубликаты в не-ID колонке
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_suspicious_id_duplicates установлен в False
    assert flags["has_suspicious_id_duplicates"] is False
    
    # Проверяем, что список подозрительных колонок пуст
    assert len(flags["suspicious_id_columns"]) == 0


def test_quality_score_with_new_heuristics():
    """Тест влияния новых эвристик на общий скор качества."""
    # ДОБАВЛЕНО: Тест влияния новых эвристик на скор качества
    
    # DataFrame с проблемами качества
    df_bad = pd.DataFrame({
        "user_id": [1, 1, 2, 2],  # дубликаты ID
        "constant_col": ["same", "same", "same", "same"],  # константная колонка
        "normal_col": [10, 20, 30, 40]
    })
    
    # DataFrame без проблем
    df_good = pd.DataFrame({
        "user_id": [1, 2, 3, 4],  # уникальные ID
        "category": ["A", "B", "C", "D"],  # разные значения
        "value": [10, 20, 30, 40]
    })
    
    # Проверяем плохой DataFrame
    summary_bad = summarize_dataset(df_bad)
    missing_bad = missing_table(df_bad)
    flags_bad = compute_quality_flags(summary_bad, missing_bad)
    
    # Проверяем хороший DataFrame
    summary_good = summarize_dataset(df_good)
    missing_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_good)
    
    # Скор качества должен быть ниже для DataFrame с проблемами
    assert flags_bad["quality_score"] < flags_good["quality_score"]
    
    # Проверяем, что проблемы обнаружены
    assert flags_bad["has_constant_columns"] is True
    assert flags_bad["has_suspicious_id_duplicates"] is True
    
    # Проверяем, что в хорошем DataFrame проблем нет
    assert flags_good["has_constant_columns"] is False
    assert flags_good["has_suspicious_id_duplicates"] is False