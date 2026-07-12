import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(row: pd.Series, names: Iterable[str], default=None):
    """Возвращает первое найденное значение по одному из возможных имён."""
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None:
                return value
    return default


def _safe_float(value, default=None):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    numeric = _safe_float(value, None)
    if numeric is None:
        return default
    return int(numeric)


def _safe_bool(value) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.number)):
        return float(value) != 0

    text = str(value).strip().lower()

    if text in {
        "", "0", "нет", "false", "no", "ложь",
        "отсутствует", "nan", "none",
    }:
        return False

    return True


def _json_safe(value):
    """Преобразует numpy/pandas-типы в значения, сериализуемые в JSON."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    return value


class ResponseService:

    @staticmethod
    def save_result(data: Dict[str, Any], file_id: str) -> str:
        file_path = RESULTS_DIR / f"{file_id}.json"

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return str(file_path)

    @staticmethod
    def get_result(file_id: str) -> Optional[Dict[str, Any]]:
        file_path = RESULTS_DIR / f"{file_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def prepare_response(
        df: pd.DataFrame,
        extra_columns: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        extra_columns:
            Признаки ручного скоринга, которые необходимо вернуть во фронт.
            Поэтому добавленный признак попадёт в Excel, а удалённый — нет.
        """
        clients = []
        extra_columns = extra_columns or []

        for position, (_, row) in enumerate(df.iterrows(), start=1):
            probability = _safe_float(
                _first_existing(
                    row,
                    [
                        "final_probability",
                        "Final_Probability",
                        "CatBoost_Probability",
                    ],
                    0,
                ),
                0.0,
            )

            risk_level = str(
                _first_existing(
                    row,
                    ["risk_level", "Risk_Level"],
                    "Низкий",
                )
            )

            completeness = _safe_float(
                _first_existing(
                    row,
                    [
                        "feature_completeness",
                        "Feature_Completeness",
                    ],
                    0,
                ),
                0.0,
            )

            top_factors = _first_existing(
                row,
                ["top_factors", "Top_Factors"],
                [],
            )

            has_competitor_source = _first_existing(
                row,
                [
                    "has_competitor",
                    "Has_Competitor",
                    "Конкурент",
                ],
                None,
            )

            has_threat_source = _first_existing(
                row,
                [
                    "has_threat",
                    "Has_Threat",
                    "Была угроза отключения",
                    "Была угроза откл",
                ],
                None,
            )

            client = {
                "client_id": position,
                "company_name": (
                    str(row.get("Контрагент"))
                    if pd.notna(row.get("Контрагент", None))
                    else ""
                ),
                "code_to": _safe_int(row.get("Код ТО", None)),
                "life_span": _safe_float(
                    row.get("Срок жизни от регистрации", None)
                ),
                "number_os": _safe_float(
                    row.get("Количество ОС", None)
                ),
                "summa_scheta_io": _safe_float(
                    row.get("Сумма счета ИО", None)
                ),
                "discount": _safe_float(
                    row.get("%Скидки", None)
                ),
                "type_of_activity": (
                    str(row.get("Вид деятельности"))
                    if pd.notna(row.get("Вид деятельности", None))
                    else None
                ),
                "annual_revenue": _safe_float(
                    row.get("Годовая выручка", None)
                ),
                "number_of_employees": _safe_float(
                    row.get("Численность сотрудников", None)
                ),
                "competitor": (
                    str(row.get("Конкурент"))
                    if pd.notna(row.get("Конкурент", None))
                    else None
                ),
                "has_competitor": _safe_bool(
                    has_competitor_source
                ),
                "has_threat": _safe_bool(
                    has_threat_source
                ),
                "total_session_minutes": _safe_float(
                    row.get("Время всех сессий в мин", None)
                ),
                "risk_score": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            _safe_float(
                                row.get("risk_score", 0),
                                0.0,
                            ),
                        ),
                    ),
                    4,
                ),
                "value_score": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            _safe_float(
                                row.get("value_score", 0),
                                0.0,
                            ),
                        ),
                    ),
                    4,
                ),
                "final_probability": round(
                    max(0.0, min(1.0, probability)),
                    4,
                ),
                "risk_level": risk_level,
                "feature_completeness": round(
                    max(0.0, min(1.0, completeness)),
                    4,
                ),
                "top_factors": _json_safe(top_factors),
            }

            # Добавляем только признаки, которые выбраны сейчас.
            # Если пользователь удалил признак, его имени не будет
            # в extra_columns, поэтому он не попадёт в ответ и Excel.
            for column_name in extra_columns:
                if column_name in row.index:
                    client[column_name] = _json_safe(
                        row.get(column_name)
                    )

            clients.append(client)

        probabilities = [
            client["final_probability"]
            for client in clients
        ]

        summary = {
            "total_clients": len(clients),
            "high_risk": sum(
                probability >= 0.6
                for probability in probabilities
            ),
            "avg_risk": (
                round(
                    sum(probabilities) / len(probabilities),
                    4,
                )
                if probabilities
                else 0
            ),
            "avg_risk_score": (
                round(
                    sum(client["risk_score"] for client in clients)
                    / len(clients),
                    4,
                )
                if clients
                else 0
            ),
            "avg_value_score": (
                round(
                    sum(client["value_score"] for client in clients)
                    / len(clients),
                    4,
                )
                if clients
                else 0
            ),
            "risk_distribution": {
                "Низкий": sum(
                    probability < 0.3
                    for probability in probabilities
                ),
                "Средний": sum(
                    0.3 <= probability < 0.6
                    for probability in probabilities
                ),
                "Высокий": sum(
                    0.6 <= probability < 0.8
                    for probability in probabilities
                ),
                "Критический": sum(
                    probability >= 0.8
                    for probability in probabilities
                ),
            },
        }

        return {
            "summary": summary,
            "clients": clients,
        }
