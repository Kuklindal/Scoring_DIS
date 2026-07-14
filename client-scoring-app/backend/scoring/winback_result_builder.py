from scoring.result_builder import get_risk_level, get_recommendation
from scoring.winback_explanation import get_winback_top_factors


def build_winback_result(df):
    df = df.copy()

    df["Final_Probability"] = (
        df["Winback_Probability"] * df["Feature_Completeness"]
    )

    df["Risk_Level"] = df["Final_Probability"].apply(get_risk_level)
    df["Recommendation"] = df["Risk_Level"].apply(get_recommendation)
    df["Top_Factors"] = df.apply(get_winback_top_factors, axis=1)

    df = df.sort_values("Final_Probability", ascending=False)

    return df
