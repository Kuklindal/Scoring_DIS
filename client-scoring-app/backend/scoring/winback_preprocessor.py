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


def reason_to_multiplier(series: pd.Series) -> pd.Series:
    """
    Причина отключения -> числовой множитель (Reason_Multiplier).

    Переиспользует REASON_MULTIPLIERS из ExcelService (тот же
    справочник, что и в формульном скоринге), чтобы не дублировать
    бизнес-логику весов причин. Пустая причина -> 0.0, неизвестная
    причина -> 1.0 (нейтральный множитель).
    """

    def get_weight(reason):
        if pd.isna(reason):
            return 0.0
        return REASON_MULTIPLIERS.get(reason, 1.0)

    return series.apply(get_weight).astype(float)


def os_step_score(series: pd.Series) -> pd.Series:
    """
    Ступенчатая шкала по количеству ОС (как в FORMULA_OPTIONS
    фронта: "Ступенчатая (ОС: 1=30, 2=70, 3+=100)").
    """

    numeric = pd.to_numeric(series, errors="coerce").fillna(0)

    def step(value: float) -> float:
        if value <= 0:
            return 0.0
        if value == 1:
            return 30.0
        if value == 2:
            return 70.0
        return 100.0

    return numeric.apply(step).astype(float)


def preprocess_winback_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Приводим названия колонок к единому виду
    df = df.rename(columns={
        "Была угроза отключения": "Была угроза откл",
        "Количество_ОС_Нормализация": "Количество_ОС_Н"
    })

    # Конкурент/угроза -> бинарные признаки, затем инверсия
    # (модель обучалась на Has_Threat_Inv/Has_Competitor_Inv).
    if "Конкурент" in df.columns:
        has_competitor = df["Конкурент"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip().lower() not in ["нет", "nan", ""] else 0
        )
    else:
        has_competitor = pd.Series(0, index=df.index)

    if "Была угроза откл" in df.columns:
        has_threat = df["Была угроза откл"].apply(
            lambda x: 1 if str(x).strip().lower() == "да" else 0
        )
    else:
        has_threat = pd.Series(0, index=df.index)

    df["Has_Competitor_Inv"] = 1.0 - has_competitor.astype(float)
    df["Has_Threat_Inv"] = 1.0 - has_threat.astype(float)

    # Причина отключения -> числовой множитель (Reason_Multiplier).
    # Сама колонка "Причина отключения" тоже остаётся в df как есть
    # (строкой) - модель использует её отдельно, как нативный
    # категориальный признак CatBoost.
    if "Причина отключения" in df.columns:
        df["Reason_Multiplier"] = reason_to_multiplier(df["Причина отключения"])
    else:
        df["Reason_Multiplier"] = 0.0
        df["Причина отключения"] = None

    if "Количество ОС" in df.columns:
        df["OS_Step_Score"] = os_step_score(df["Количество ОС"])
    else:
        df["OS_Step_Score"] = 0.0

    if "Вид деятельности" not in df.columns:
        df["Вид деятельности"] = None

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
