import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class ResponseService:
    @staticmethod
    def save_result(data: Dict[str, Any], file_id: str) -> str:
        file_path = RESULTS_DIR / f"{file_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(file_path)

    @staticmethod
    def get_result(file_id: str) -> Dict[str, Any]:
        file_path = RESULTS_DIR / f"{file_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def prepare_response(df: pd.DataFrame) -> Dict[str, Any]:
        clients = []
        for idx, row in df.iterrows():
            client = {
                "client_id": idx + 1,
                "company_name": str(row.get("Контрагент", "")),
                "code_to": int(row.get("Код ТО", 0)) if pd.notna(row.get("Код ТО", 0)) else None,
                "life_span": float(row.get("Срок жизни от регистрации", 0)) if pd.notna(
                    row.get("Срок жизни от регистрации", 0)) else None,
                "number_os": float(row.get("Количество ОС", 0)) if pd.notna(row.get("Количество ОС", 0)) else None,
                "summa_scheta_io": float(row.get("Сумма счета ИО", 0)) if pd.notna(
                    row.get("Сумма счета ИО", 0)) else None,
                "discount": float(row.get("%Скидки", 0)) if pd.notna(row.get("%Скидки", 0)) else None,
                "type_of_activity": str(row.get("Вид деятельности", "")) if pd.notna(
                    row.get("Вид деятельности", "")) else None,
                "annual_revenue": float(row.get("Годовая выручка", 0)) if pd.notna(
                    row.get("Годовая выручка", 0)) else None,
                "number_of_employees": float(row.get("Численность сотрудников", 0)) if pd.notna(
                    row.get("Численность сотрудников", 0)) else None,
                "competitor": str(row.get("Конкурент", "")) if pd.notna(row.get("Конкурент", "")) else None,
                "has_competitor": pd.notna(row.get("Конкурент", None)),
                "has_threat": pd.notna(row.get("Была угроза отключения", None)),
                "total_session_minutes": float(row.get("Время всех сессий в мин", 0)) if pd.notna(
                    row.get("Время всех сессий в мин", 0)) else None,
                "final_probability": round(float(row.get("final_probability", 0)), 4),
                "risk_level": str(row.get("risk_level", "Низкий")),
                "feature_completeness": round(float(row.get("feature_completeness", 0)), 4),
                "top_factors": row.get("top_factors", [])
            }
            clients.append(client)

        probs = [c["final_probability"] for c in clients]
        summary = {
            "total_clients": len(clients),
            "high_risk": len([p for p in probs if p >= 0.6]),
            "avg_risk": round(sum(probs) / len(probs), 4) if probs else 0,
            "risk_distribution": {
                "Низкий": len([p for p in probs if p < 0.3]),
                "Средний": len([p for p in probs if 0.3 <= p < 0.6]),
                "Высокий": len([p for p in probs if 0.6 <= p < 0.8]),
                "Критический": len([p for p in probs if p >= 0.8])
            }
        }
        return {"summary": summary, "clients": clients}