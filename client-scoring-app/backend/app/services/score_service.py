import pandas as pd
import numpy as np
from typing import List
from app.services.excel_service import ExcelService
from app.services.formula_service import FormulaService
from app.services.excel_service import RISK_LEVELS


class ScoreService:

    @staticmethod
    def calculate_auto_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "winback" and "Причина отключения" not in df.columns:
            df["Причина отключения"] = None

        result = ExcelService.calculate_scores(df)
        return result

    @staticmethod
    def calculate_manual_score(df: pd.DataFrame, features: list) -> pd.DataFrame:
        result = df.copy()
        total_score = pd.Series(0.0, index=df.index)

        for feature in features:
            col = result[feature.name]
            normalized = FormulaService.apply_formula(col, feature.formula, feature.type)
            total_score += normalized * feature.weight

        final_prob = np.clip(total_score, 0, 1)

        def get_risk(p):
            for low, high, level in RISK_LEVELS:
                if low <= p < high:
                    return level
            return "Критический"

        result["final_probability"] = final_prob
        result["risk_level"] = final_prob.apply(get_risk)

        existing_cols = [c for c in df.columns if c in result.columns]
        if existing_cols:
            result["feature_completeness"] = result[existing_cols].notna().sum(axis=1) / len(existing_cols)
        else:
            result["feature_completeness"] = 1.0

        result["top_factors"] = result.apply(lambda row: [], axis=1)

        return result