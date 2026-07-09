from scoring.explanation import get_top_factors


def get_risk_level(probability):
    if probability >= 0.7:
        return "Высокий"
    elif probability >= 0.4:
        return "Средний"
    return "Низкий"


def get_recommendation(risk_level):
    if risk_level == "Высокий":
        return "Связаться с клиентом в течение 3 дней"
    elif risk_level == "Средний":
        return "Поставить клиента на контроль"
    return "Плановое сопровождение"


def build_result(df):
    df = df.copy()

    df["Final_Probability"] = (
        df["CatBoost_Probability"] * df["Feature_Completeness"]
    )

    df["Risk_Level"] = df["Final_Probability"].apply(get_risk_level)
    df["Recommendation"] = df["Risk_Level"].apply(get_recommendation)
    df["Top_Factors"] = df.apply(get_top_factors, axis=1)

    df = df.sort_values("Final_Probability", ascending=False)

    return df