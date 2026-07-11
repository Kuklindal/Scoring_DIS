import pandas as pd


class FormulaService:
    """
    Применяется к УЖЕ очищенной колонке (см. DataCleaningService.clean_column).
    Т.е. на входе всегда числовой pd.Series:
    - numeric/percent -> реальные числа (могут быть NaN)
    - binary -> уже 0.0 / 1.0 / NaN
    """

    @staticmethod
    def apply_formula(series: pd.Series, formula: str, feature_type: str) -> pd.Series:
        if formula == "linear":
            s = series.fillna(0)
            min_val, max_val = s.min(), s.max()
            if max_val == min_val:
                return pd.Series(0.5, index=s.index)
            return (s - min_val) / (max_val - min_val)

        elif formula == "inverse":
            s = series.fillna(0)
            min_val, max_val = s.min(), s.max()
            if max_val == min_val:
                return pd.Series(0.5, index=s.index)
            return 1 - (s - min_val) / (max_val - min_val)

        elif formula == "binary":
            # ВАЖНО: раньше здесь было series.notna().astype(float), из-за
            # чего явное "Нет"/0 после очистки (уже 0.0, не NaN) считалось
            # как 1. Данные к этому моменту уже приведены к 0/1 в
            # DataCleaningService.clean_binary, поэтому здесь просто
            # используем очищенное значение, а не факт "заполнено/не заполнено".
            return series.fillna(0).astype(float)

        elif formula == "step":
            s = series.fillna(0)

            def step_fn(x):
                if x <= 1:
                    return 0.3
                elif x == 2:
                    return 0.7
                else:
                    return 1.0

            return s.apply(step_fn)

        elif formula == "none":
            # значение уже в шкале 0-1 (например, готовая вероятность) - просто клип
            return series.fillna(0).clip(0, 1)

        else:
            s = series.fillna(0)
            min_val, max_val = s.min(), s.max()
            if max_val == min_val:
                return pd.Series(0.5, index=s.index)
            return (s - min_val) / (max_val - min_val)