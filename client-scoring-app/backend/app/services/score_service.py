import pandas as pd

from app.services.data_cleaning_service import DataCleaningService
from app.services.excel_service import RISK_LEVELS
from app.services.formula_service import FormulaService


FINAL_RISK_WEIGHT = 0.65
FINAL_VALUE_WEIGHT = 0.35


class ScoreService:

    @staticmethod
    def calculate_manual_score(
        df: pd.DataFrame,
        features: list,
    ) -> pd.DataFrame:
        result = df.copy()

        risk_score = pd.Series(
            0.0,
            index=result.index,
            dtype=float,
        )
        value_score = pd.Series(
            0.0,
            index=result.index,
            dtype=float,
        )

        risk_contributions = {}
        value_contributions = {}
        original_completeness = pd.DataFrame(index=result.index)

        for feature in features:
            feature_name = str(feature.name).strip()

            if feature_name not in result.columns:
                raise ValueError(
                    f"В файле отсутствует колонка '{feature_name}'"
                )

            raw_column = result[feature_name]

            # Полнота считается один раз по каждому уникальному признаку.
            original_completeness[feature_name] = (
                raw_column.notna()
                & (raw_column.astype(str).str.strip() != "")
            )

            cleaned = DataCleaningService.clean_column(
                raw_column,
                feature.type,
            )

            if float(feature.risk_weight) > 0:
                normalized_risk = FormulaService.apply_formula(
                    cleaned,
                    feature.formula,
                    feature.type,
                )
                normalized_risk = pd.to_numeric(
                    normalized_risk,
                    errors="coerce",
                ).fillna(0.0)

                risk_contribution = (
                    normalized_risk * float(feature.risk_weight)
                )
                risk_contributions[feature_name] = risk_contribution
                risk_score = risk_score.add(
                    risk_contribution,
                    fill_value=0.0,
                )

            if float(feature.value_weight) > 0:
                normalized_value = FormulaService.apply_formula(
                    cleaned,
                    feature.formula,
                    feature.type,
                )
                normalized_value = pd.to_numeric(
                    normalized_value,
                    errors="coerce",
                ).fillna(0.0)

                value_contribution = (
                    normalized_value * float(feature.value_weight)
                )
                value_contributions[feature_name] = value_contribution
                value_score = value_score.add(
                    value_contribution,
                    fill_value=0.0,
                )

            normalized_type = (
                DataCleaningService.normalize_feature_type(
                    feature.type
                )
            )
            if normalized_type == "binary":
                technical_name = feature_name.lower()

                if "конкурент" in technical_name:
                    result["has_competitor"] = cleaned.astype(float)

                if "угроз" in technical_name:
                    result["has_threat"] = cleaned.astype(float)

        risk_score = risk_score.clip(0.0, 1.0)
        value_score = value_score.clip(0.0, 1.0)

        final_probability = (
            risk_score * FINAL_RISK_WEIGHT
            + value_score * FINAL_VALUE_WEIGHT
        ).clip(0.0, 1.0)

        def get_risk(probability: float) -> str:
            for low, high, level in RISK_LEVELS:
                if low <= probability < high:
                    return level
            return "Критический"

        result["risk_score"] = risk_score
        result["value_score"] = value_score
        result["final_probability"] = final_probability
        result["risk_level"] = final_probability.apply(get_risk)

        if not original_completeness.empty:
            result["feature_completeness"] = (
                original_completeness.sum(axis=1)
                / len(original_completeness.columns)
            )
        else:
            result["feature_completeness"] = 1.0

        combined_contributions = {}

        for name, contribution in risk_contributions.items():
            combined_contributions[f"Риск: {name}"] = (
                contribution * FINAL_RISK_WEIGHT
            )

        for name, contribution in value_contributions.items():
            combined_contributions[f"Ценность: {name}"] = (
                contribution * FINAL_VALUE_WEIGHT
            )

        contribution_frame = pd.DataFrame(
            combined_contributions,
            index=result.index,
        )

        def get_top_factors(row_index):
            if contribution_frame.empty:
                return []

            row = (
                contribution_frame
                .loc[row_index]
                .sort_values(ascending=False)
            )
            top = row[row > 0].head(4)

            return [
                f"{name} (вклад {value:.2f})"
                for name, value in top.items()
            ]

        result["top_factors"] = [
            get_top_factors(index)
            for index in result.index
        ]

        return result
