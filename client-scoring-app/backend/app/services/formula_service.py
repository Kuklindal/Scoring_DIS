import pandas as pd


FORMULA_ALIASES = {
    "linear": "linear",
    "min_max": "linear",
    "min-max": "linear",

    "inverse": "inverse",
    "min_max_inverse": "inverse",
    "min-max inverse": "inverse",

    "binary": "binary",

    "step": "step",
    "step_os": "step",

    "none": "none",
}


class FormulaService:
    """
    Применяет формулу к уже очищенной числовой колонке.
    Поддерживает названия формул из backend и Streamlit frontend.
    """

    @staticmethod
    def normalize_formula(formula: str) -> str:
        key = str(formula or "").strip().lower()
        normalized = FORMULA_ALIASES.get(key)

        if normalized is None:
            raise ValueError(
                f"Неизвестная формула: '{formula}'. "
                "Допустимые значения: linear/min_max, "
                "inverse/min_max_inverse, binary, step/step_os, none."
            )

        return normalized

    @staticmethod
    def apply_formula(
        series: pd.Series,
        formula: str,
        feature_type: str,
    ) -> pd.Series:
        normalized_formula = FormulaService.normalize_formula(formula)
        numeric_series = pd.to_numeric(series, errors="coerce")

        if normalized_formula == "linear":
            valid = numeric_series.dropna()

            if valid.empty:
                return pd.Series(0.0, index=series.index)

            minimum = valid.min()
            maximum = valid.max()

            if maximum == minimum:
                return pd.Series(0.5, index=series.index)

            # Пропуски не должны искусственно менять min/max.
            result = (
                numeric_series - minimum
            ) / (maximum - minimum)
            return result.fillna(0.0).clip(0.0, 1.0)

        if normalized_formula == "inverse":
            valid = numeric_series.dropna()

            if valid.empty:
                return pd.Series(0.0, index=series.index)

            minimum = valid.min()
            maximum = valid.max()

            if maximum == minimum:
                return pd.Series(0.5, index=series.index)

            result = 1.0 - (
                (numeric_series - minimum)
                / (maximum - minimum)
            )
            return result.fillna(0.0).clip(0.0, 1.0)

        if normalized_formula == "binary":
            return (
                numeric_series
                .fillna(0.0)
                .clip(0.0, 1.0)
                .astype(float)
            )

        if normalized_formula == "step":
            values = numeric_series.fillna(0.0)

            def step_function(value: float) -> float:
                if value <= 0:
                    return 0.0
                if value <= 1:
                    return 0.3
                if value == 2:
                    return 0.7
                return 1.0

            return values.apply(step_function).astype(float)

        if normalized_formula == "none":
            return (
                numeric_series
                .fillna(0.0)
                .clip(0.0, 1.0)
                .astype(float)
            )

        raise ValueError(
            f"Не удалось применить формулу '{formula}'"
        )
