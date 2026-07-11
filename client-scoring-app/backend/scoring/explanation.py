def get_top_factors(row):
    factors = []

    if row.get("Has_Competitor", 0) == 1:
        factors.append("Есть конкурент")

    if row.get("Has_Threat", 0) == 1:
        factors.append("Была угроза отключения")

    if row.get("%Скидки", 0) >= 50:
        factors.append("Высокая скидка")

    if row.get("Feature_Completeness", 1) < 0.8:
        factors.append("Низкая полнота данных")

    if row.get("Время всех сессий в мин", 0) < 10000:
        factors.append("Низкая активность в системе")

    return factors[:4]