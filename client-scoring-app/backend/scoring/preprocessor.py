import pandas as pd
import numpy as np


def clean_numeric_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace(["nan", "None", ""], np.nan)
        .astype(float)
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Приводим названия колонок к единому виду
    df = df.rename(columns={
        "Была угроза отключения": "Была угроза откл",
        "Количество_ОС_Нормализация": "Количество_ОС_Н"
    })

    # Конкурент → бинарный признак
    if "Конкурент" in df.columns:
        df["Has_Competitor"] = df["Конкурент"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip().lower() not in ["нет", "nan", ""] else 0
        )
    else:
        df["Has_Competitor"] = 0

    # Была угроза отключения → бинарный признак
    if "Была угроза откл" in df.columns:
        df["Has_Threat"] = df["Была угроза откл"].apply(
            lambda x: 1 if str(x).strip().lower() == "да" else 0
        )
    else:
        df["Has_Threat"] = 0

    numeric_cols = [
        "Код ТО",
        "Срок жизни от регистрации",
        "Количество ОС",
        "Количество_ОС_Н",
        "Сумма счета ИО",
        "%Скидки",
        "Годовая выручка",
        "Численность сотрудников",
        "Время всех сессий в мин",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    return df