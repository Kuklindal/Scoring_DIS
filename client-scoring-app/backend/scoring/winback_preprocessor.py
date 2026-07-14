import pandas as pd
import numpy as np

from app.services.excel_service import REASON_MULTIPLIERS


def clean_numeric_column(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace(["nan", "None", ""], np.nan)
        .astype(float)
    )


def clean_reason_column(series: pd.Series) -> pd.Series:
    """
    Причина отключения -> числовой вес (Reason_Weight).

    Используем уже существующие коэффициенты REASON_MULTIPLIERS
    из ExcelService, чтобы не дублировать бизнес-логику весов причин.
    Пустая причина -> 0.0, неизвестная причина -> 1.0 (нейтральный вес),
    как и в ExcelService.get_reason_weight.
    """

    def get_weight(reason):
        if pd.isna(reason):
            return 0.0
        return REASON_MULTIPLIERS.get(reason, 1.0)

    return series.apply(get_weight).astype(float)


def preprocess_winback_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Приводим названия колонок к единому виду
    df = df.rename(columns={
        "Была угроза отключения": "Была угроза откл",
        "Количество_ОС_Нормализация": "Количество_ОС_Н"
    })

    # Конкурент -> бинарный признак
    if "Конкурент" in df.columns:
        df["Has_Competitor"] = df["Конкурент"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip().lower() not in ["нет", "nan", ""] else 0
        )
    else:
        df["Has_Competitor"] = 0

    # Была угроза отключения -> бинарный признак
    if "Была угроза откл" in df.columns:
        df["Has_Threat"] = df["Была угроза откл"].apply(
            lambda x: 1 if str(x).strip().lower() == "да" else 0
        )
    else:
        df["Has_Threat"] = 0

    # Причина отключения -> числовой вес (есть только у ушедших клиентов)
    if "Причина отключения" in df.columns:
        df["Reason_Weight"] = clean_reason_column(df["Причина отключения"])
    else:
        df["Reason_Weight"] = 1.0

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
