import numpy as np
import pandas as pd

from scoring.expert_score import calculate_manual_risk_score


FEATURE_COLUMNS = [
    "Срок жизни от регистрации",
    "Количество ОС",
    "Сумма счета ИО",
    "%Скидки",
    "Годовая выручка",
    "Численность сотрудников",
    "Has_Competitor",
    "Has_Threat",
    "Время всех сессий в мин",
    "Manual_Risk_Score",
]


COMPLETENESS_FEATURES = [
    "Код ТО",
    "Срок жизни от регистрации",
    "Количество ОС",
    "Сумма счета ИО",
    "%Скидки",
    "Вид деятельности",
    "Годовая выручка",
    "Численность сотрудников",
    "Конкурент",
    "Была угроза откл",
    "Время всех сессий в мин",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Manual_Risk_Score"] = calculate_manual_risk_score(df)

    existing_completeness_features = [
        col for col in COMPLETENESS_FEATURES if col in df.columns
    ]

    df["Feature_Completeness"] = (
        df[existing_completeness_features].notna().sum(axis=1)
        / len(existing_completeness_features)
    )

    if "Сумма счета ИО" in df.columns:
        df["Log_Invoice_Amount"] = np.log1p(df["Сумма счета ИО"])

    if "Годовая выручка" in df.columns:
        df["Log_Annual_Revenue"] = np.log1p(df["Годовая выручка"])

    return df