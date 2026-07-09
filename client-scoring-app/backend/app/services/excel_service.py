import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path

REQUIRED_COLUMNS_CHURN = [
    "Контрагент", "Код ТО", "Срок жизни от регистрации", "Количество ОС",
    "Сумма счета ИО", "%Скидки", "Вид деятельности", "Годовая выручка",
    "Численность сотрудников", "Конкурент", "Была угроза отключения",
    "Время всех сессий в мин"
]

REQUIRED_COLUMNS_WINBACK = [
    "Контрагент", "Код ТО", "Срок жизни от регистрации", "Количество ОС",
    "Сумма счета ИО", "%Скидки", "Вид деятельности", "Годовая выручка",
    "Численность сотрудников", "Конкурент", "Была угроза отключения",
    "Время всех сессий в мин", "Причина отключения"
]

WEIGHTS = {
    "life_span": 0.075, "number_os": 0.10, "summa_scheta_io": 0.15,
    "discount": 0.075, "annual_revenue": 0.10, "number_of_employees": 0.05,
    "total_session_minutes": 0.15, "competitor": 0.10, "has_threat": 0.05, "reason": 0.15
}

REASON_MULTIPLIERS = {
    "Ликвидация / Клиент пропал": 0.5, "ДОЛГИ (инициатива РИЦ)": 0.5,
    "Нет денег для оплаты счетов по К+": 0.75, "Есть наш К+ на другом ЮЛ или ТО": 0.75,
    "К+ от другого РИЦ": 0.75, "Экономят, сокращают расходы": 1.0,
    "Нет потребности в К+/сервисе": 1.0, "Конкуренты (кроме других РИЦ)": 1.0,
    "Есть предоплата. Отключение на короткий период": 1.0,
    "Правило1 по комплектам": 1.0, "НАВЕС": 1.0, "Изменение комплекта": 1.0
}

RISK_LEVELS = [
    (0.0, 0.3, "Низкий"), (0.3, 0.6, "Средний"),
    (0.6, 0.8, "Высокий"), (0.8, 1.0, "Критический")
]


class ExcelService:
    @staticmethod
    def validate_file(file_path: str, mode: str):
        errors = []

        try:
            if mode == "churn":
                df = pd.read_excel(file_path, sheet_name="КИС_дляСкорингаДИС_Сопр", header=0)
                required_columns = REQUIRED_COLUMNS_CHURN
            else:
                df = pd.read_excel(file_path, sheet_name="КИС_дляСкорингаДИС_Откл", header=0)
                required_columns = REQUIRED_COLUMNS_WINBACK

        except Exception as e:
            errors.append(f"Ошибка чтения файла: {str(e)}")
            return False, errors, None

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            errors.append(f"Отсутствуют столбцы: {', '.join(missing)}")

        if df.empty:
            errors.append("Файл не содержит данных")

        return len(errors) == 0, errors, df

    @staticmethod
    def normalize_series(series: pd.Series) -> pd.Series:
        series = series.fillna(0)
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def invert_normalize(series: pd.Series) -> pd.Series:
        return 1 - ExcelService.normalize_series(series)

    @staticmethod
    def get_reason_weight(reason) -> float:
        return 0 if pd.isna(reason) else REASON_MULTIPLIERS.get(reason, 1.0)

    @staticmethod
    def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        # Нормализация
        norm = {
            "life": ExcelService.normalize_series(result["Срок жизни от регистрации"]),
            "os": ExcelService.normalize_series(result["Количество ОС"]),
            "invoice": ExcelService.normalize_series(result["Сумма счета ИО"]),
            "discount": ExcelService.invert_normalize(result["%Скидки"]),
            "revenue": ExcelService.normalize_series(result["Годовая выручка"]),
            "employees": ExcelService.normalize_series(result["Численность сотрудников"]),
            "sessions": ExcelService.normalize_series(result["Время всех сессий в мин"]),
            "competitor": result["Конкурент"].isna().astype(float),
            "threat": result["Была угроза отключения"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["да", "есть", "true", "1"])
                        .astype(float),
        }
        if "Причина отключения" not in result.columns:
            result["Причина отключения"] = None
        reason_w = result["Причина отключения"].apply(ExcelService.get_reason_weight)

        # Взвешенная сумма
        weighted = (
                norm["life"] * WEIGHTS["life_span"] +
                norm["os"] * WEIGHTS["number_os"] +
                norm["invoice"] * WEIGHTS["summa_scheta_io"] +
                norm["discount"] * WEIGHTS["discount"] +
                norm["revenue"] * WEIGHTS["annual_revenue"] +
                norm["employees"] * WEIGHTS["number_of_employees"] +
                norm["sessions"] * WEIGHTS["total_session_minutes"] +
                norm["competitor"] * WEIGHTS["competitor"] +
                norm["threat"] * WEIGHTS["has_threat"] +
                reason_w * WEIGHTS["reason"]
        )

        final_prob = np.clip(weighted * 100, 0, 100) / 100
        result["final_probability"] = final_prob

        def get_risk(p):
            for low, high, level in RISK_LEVELS:
                if low <= p < high:
                    return level
            return "Критический"

        result["risk_level"] = final_prob.apply(get_risk)

        cols = ["Срок жизни от регистрации", "Количество ОС", "Сумма счета ИО",
                "%Скидки", "Годовая выручка", "Численность сотрудников",
                "Время всех сессий в мин", "Причина отключения"]
        
        result["feature_completeness"] = result[cols].notna().sum(axis=1) / len(cols)

        def get_top(row):
            factors = []
            if pd.notna(row.get("Конкурент")):
                factors.append("Есть конкурент")
            if pd.notna(row.get("Была угроза отключения")):
                factors.append("Была угроза")
            if pd.notna(row.get("Причина отключения")):
                factors.append(f"Причина: {row['Причина отключения']}")
            if row.get("%Скидки", 0) > 20:
                factors.append("Высокая скидка")
            if row.get("Время всех сессий в мин", 999999) < 100:
                factors.append("Низкая активность")
            if row.get("Срок жизни от регистрации", 999) < 2:
                factors.append("Короткий срок жизни")
            return factors[:3]

        result["top_factors"] = result.apply(get_top, axis=1)
        return result