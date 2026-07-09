import pandas as pd


def calculate_manual_risk_score(df: pd.DataFrame) -> pd.Series:
    scores = []

    max_session = df["Время всех сессий в мин"].max()
    min_session = df["Время всех сессий в мин"].min()

    max_life = df["Срок жизни от регистрации"].max()
    min_life = df["Срок жизни от регистрации"].min()

    for _, row in df.iterrows():
        score = 0

        if row.get("Has_Competitor", 0) == 1:
            score += 25

        if row.get("Has_Threat", 0) == 1:
            score += 25

        if pd.notna(row.get("Время всех сессий")) and max_session != min_session:
            norm_session = (
                row["Время всех сессий в мин"] - min_session
            ) / (max_session - min_session)

            score += (1 - norm_session) * 20

        if pd.notna(row.get("Срок жизни от регистрации")) and max_life != min_life:
            norm_life = (
                row["Срок жизни от регистрации"] - min_life
            ) / (max_life - min_life)

            score += (1 - norm_life) * 15

        scores.append(round(score, 2))

    return pd.Series(scores, index=df.index)