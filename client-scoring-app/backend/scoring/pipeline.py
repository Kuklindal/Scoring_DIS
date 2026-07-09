import pandas as pd

from scoring.preprocessor import preprocess_data
from scoring.feature_builder import build_features
from scoring.model_service import predict_churn_probability
from scoring.result_builder import build_result


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_data(df)

    df = build_features(df)

    df["CatBoost_Probability"] = predict_churn_probability(df)

    result_df = build_result(df)

    return result_df