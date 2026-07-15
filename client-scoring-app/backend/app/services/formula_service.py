import pandas as pd

from app.services.excel_service import REASON_MULTIPLIERS


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

    # Добавлено для скоринга возвращаемости (winback):
    "binary_inverse": "binary_inverse",
    "reason_multiplier": "reason_multiplier",
    "categorical": "categorical",
}


class FormulaService:
    """
    Применяет формулу к уже очищенной колонке.
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
                "inverse/min_max_inverse, binary, binary_inverse, "
                "step/step_os, reason_multiplier, categorical, none."
            )

        return normalized

    @staticmethod
    def _apply_reason_multiplier(series: pd.Series) -> pd.Series:
        """
        Причина отключения -> вес возврата (0..1).

        Переиспользует уже существующий REASON_MULTIPLIERS
        (тот же справочник, что и в ExcelService), чтобы не
        дублировать бизнес-логику весов причин отключения.
        Пустая причина -> 0.0, неизвестная причина -> 1.0
        (нейтральный вес, как и в ExcelService.get_reason_weight).
        """

        def get_weight(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 0.0
            return REASON_MULTIPLIERS.get(value, 1.0)

        return series.apply(get_weight).astype(float).clip(0.0, 1.0)

    @staticmethod
    def _apply_categorical_presence(series: pd.Series) -> pd.Series:
        """
        Общий категориальный признак без справочника весов:
        просто "заполнено / не заполнено", аналогично binary,
        но без приведения текста к 0/1 по словам "да"/"нет".
        """

        def get_presence(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 0.0
            return 1.0 if str(value).strip() else 0.0

        return series.apply(get_presence).astype(float)

    @staticmethod
    def apply_formula(
        series: pd.Series,
        formula: str,
        feature_type: str,
    ) -> pd.Series:
        normalized_formula = FormulaService.normalize_formula(formula)

        # Эти две формулы работают с исходным текстом (категориальные
        # значения), поэтому обрабатываются до pd.to_numeric ниже -
        # иначе строки превратились бы в NaN.
        if normalized_formula == "reason_multiplier":
            return FormulaService._apply_reason_multiplier(series)

        if normalized_formula == "categorical":
            return FormulaService._apply_categorical_presence(series)

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

        if normalized_formula == "binary_inverse":
            return (
                1.0
                - numeric_series
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
