from scoring.explanation import get_top_factors


def get_winback_top_factors(row):
    """
    Дополняет базовые факторы (scoring.explanation.get_top_factors)
    причиной отключения - она есть только у ушедших клиентов.
    """
    factors = get_top_factors(row)

    reason = row.get("Причина отключения")
    if reason is not None and str(reason).strip().lower() not in ("", "nan", "none"):
        factors.append(f"Причина отключения: {reason}")

    return factors[:4]
