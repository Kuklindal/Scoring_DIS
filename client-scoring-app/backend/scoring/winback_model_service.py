import pandas as pd
from catboost import CatBoostClassifier, Pool

from scoring.winback_feature_builder import (
    FEATURE_COLUMNS_WINBACK,
    CAT_FEATURES_WINBACK,
)


MODEL_PATH_WINBACK = "models/winback_model.cbm"


def load_winback_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH_WINBACK)
    return model


def predict_winback_probability(df: pd.DataFrame) -> pd.Series:
    model = load_winback_model()

    X = df[FEATURE_COLUMNS_WINBACK]

    # Категориальные признаки (индексы 11, 12 - "Вид деятельности" и
    # "Причина отключения") передаются через Pool с явным указанием
    # cat_features, иначе CatBoost не поймёт, что это не числа.
    pool = Pool(X, cat_features=CAT_FEATURES_WINBACK)

    probabilities = model.predict_proba(pool)[:, 1]

    return pd.Series(probabilities, index=df.index)
