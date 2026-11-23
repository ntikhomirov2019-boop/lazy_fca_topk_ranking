"""Загрузка датасета German Credit (Statlog, UCI).

Файл german.data: 1000 строк, 21 колонка (20 признаков + target),
разделитель — пробел, заголовка нет.
Целевая переменная в исходнике: 1 = good, 2 = bad.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMNS = [
    "checking_status",        # 1  категориальный
    "duration",               # 2  числовой
    "credit_history",         # 3  категориальный
    "purpose",                # 4  категориальный
    "credit_amount",          # 5  числовой
    "savings_status",         # 6  категориальный
    "employment",             # 7  категориальный
    "installment_commitment", # 8  числовой
    "personal_status",        # 9  категориальный
    "other_parties",          # 10 категориальный
    "residence_since",        # 11 числовой
    "property_magnitude",     # 12 категориальный
    "age",                    # 13 числовой
    "other_payment_plans",    # 14 категориальный
    "housing",                # 15 категориальный
    "existing_credits",       # 16 числовой
    "job",                    # 17 категориальный
    "num_dependents",         # 18 числовой
    "own_telephone",          # 19 категориальный
    "foreign_worker",         # 20 категориальный
    "class",                  # 21 target: 1=good, 2=bad
]

NUMERIC_COLUMNS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]

CATEGORICAL_COLUMNS = [c for c in COLUMNS if c not in NUMERIC_COLUMNS + ["class"]]


def load_german_credit(path: str | Path) -> pd.DataFrame:
    """Загружает German Credit из raw-файла UCI.

    Parameters
    ----------
    path : str | Path
        Путь к german.data, например "data/raw/german.data".

    Returns
    -------
    pd.DataFrame
        Таблица 1000 × 21:
        - колонки переименованы в читаемые имена;
        - числовые признаки приведены к int;
        - категориальные оставлены как str (коды A11, A12, ...);
        - `class` закодирован как int: 1 = good, 0 = bad.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Скачайте german.data с UCI "
            f"(https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) "
            f"и положите в data/raw/."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].astype(int)
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].astype(str)

    if not df["class"].isin([1, 2]).all():
        raise ValueError(
            f"Ожидались значения target 1/2, найдены: {sorted(df['class'].unique())}"
        )
    df["class"] = df["class"].map({1: 1, 2: 0}).astype(int)

    return df