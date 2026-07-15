import pandas as pd

from scoring.winback_preprocessor import preprocess_winback_data
from scoring.winback_feature_builder import build_winback_features
from scoring.winback_model_service import predict_winback_probability
from scoring.winback_result_builder import build_winback_result


def calculate_winback_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_winback_data(df)

    df = build_winback_features(df)

    df["Winback_Probability"] = predict_winback_probability(df)

    result_df = build_winback_result(df)

    return result_df
