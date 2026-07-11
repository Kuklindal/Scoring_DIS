import pandas as pd
import numpy as np
from typing import List

from app.services.excel_service import ExcelService, RISK_LEVELS
from app.services.formula_service import FormulaService
from app.services.data_cleaning_service import DataCleaningService


class ScoreService:

    @staticmethod
    def calculate_auto_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "winback" and "Причина отключения" not in df.columns:
            df["Причина отключения"] = None

        result = ExcelService.calculate_scores(df)
        return result

    @staticmethod
    def calculate_manual_score(df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        features: List[FeatureConfig] (name, weight, type, formula - формула
        уже дефолтизирована на уровне схемы, если не была указана явно).
        """
        result = df.copy()
        total_score = pd.Series(0.0, index=df.index)

        # per-feature: (normalized_series, weight) - нужно для top_factors
        contributions = {}

        for feature in features:
            raw_col = result[feature.name]
            cleaned = DataCleaningService.clean_column(raw_col, feature.type)
            normalized = FormulaService.apply_formula(cleaned, feature.formula, feature.type)
            contributions[feature.name] = normalized * feature.weight
            total_score += normalized * feature.weight

        final_prob = np.clip(total_score, 0, 1)

        def get_risk(p):
            for low, high, level in RISK_LEVELS:
                if low <= p < high:
                    return level
            return "Критический"

        result["final_probability"] = final_prob
        result["risk_level"] = final_prob.apply(get_risk)

        feature_names = [f.name for f in features]
        existing_cols = [c for c in feature_names if c in result.columns]
        if existing_cols:
            result["feature_completeness"] = result[existing_cols].notna().sum(axis=1) / len(existing_cols)
        else:
            result["feature_completeness"] = 1.0

        # top_factors: 3 признака с наибольшим вкладом (вес * нормализованное значение) в этой строке
        contrib_df = pd.DataFrame(contributions, index=result.index)

        def get_top(row_idx):
            if contrib_df.empty:
                return []
            row = contrib_df.loc[row_idx].sort_values(ascending=False)
            top = row[row > 0].head(3)
            return [f"{name} (вклад {value:.2f})" for name, value in top.items()]

        result["top_factors"] = [get_top(idx) for idx in result.index]

        return result