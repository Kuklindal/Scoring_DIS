import os
import pandas as pd
from catboost import CatBoostClassifier

from scoring.feature_builder import FEATURE_COLUMNS


MODEL_PATH = "models/churn_model.cbm"


def train_model(df_all: pd.DataFrame) -> CatBoostClassifier:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X = df_all[FEATURE_COLUMNS]
    y = df_all["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=100,
        random_seed=42
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC AUC: {roc_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)

    return model


def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_churn_probability(df: pd.DataFrame) -> pd.Series:
    model = load_model()

    X = df[FEATURE_COLUMNS]

    probabilities = model.predict_proba(X)[:, 1]

    return pd.Series(probabilities, index=df.index)