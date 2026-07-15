import numpy as np
import pandas as pd


# ВАЖНО: порядок и имена ниже - это ТОЧНЫЙ список признаков, с которым
# обучалась модель winback_model.cbm (восстановлен из бинарника модели,
# см. переписку/анализ). Индексы 11 и 12 - категориальные (сырые
# строки), передаются в CatBoost как native categorical features.
FEATURE_COLUMNS_WINBACK = [
    "Data_Completeness",
    "Reason_Multiplier",
    "Время всех сессий в мин",
    "Has_Threat_Inv",
    "Has_Competitor_Inv",
    "Численность сотрудников",
    "Годовая выручка",
    "%Скидки",
    "Сумма счета ИО",
    "OS_Step_Score",
    "Срок жизни от регистрации",
    "Вид деятельности",
    "Причина отключения",
]

# Индексы (в FEATURE_COLUMNS_WINBACK) признаков, которые модель
# обучалась воспринимать как категориальные (сырой текст, без
# числового кодирования).
CAT_FEATURES_WINBACK = [11, 12]

COMPLETENESS_FEATURES_WINBACK = [
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
    "Причина отключения",
]


def build_winback_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_completeness_features = [
        col for col in COMPLETENESS_FEATURES_WINBACK if col in df.columns
    ]

    # Data_Completeness - то же самое, что Feature_Completeness в
    # остальном коде проекта, но под именем, на котором обучалась
    # модель.
    df["Data_Completeness"] = (
        df[existing_completeness_features].notna().sum(axis=1)
        / len(existing_completeness_features)
    )
    df["Feature_Completeness"] = df["Data_Completeness"]

    # Категориальные признаки для CatBoost должны остаться строками
    # (не NaN-float) - пустые значения заменяем на явный текст, как
    # это принято в CatBoost native categorical handling.
    for cat_column in ("Вид деятельности", "Причина отключения"):
        df[cat_column] = (
            df[cat_column]
            .astype("object")
            .where(df[cat_column].notna(), "не указано")
            .astype(str)
        )

    for column in FEATURE_COLUMNS_WINBACK:
        if column not in df.columns:
            raise ValueError(
                f"Не удалось построить признак модели: '{column}' "
                "отсутствует после препроцессинга"
            )

    return df
