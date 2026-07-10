import pandas as pd


class FormulaService:
    @staticmethod
    def apply_formula(series: pd.Series, formula: str, feature_type: str) -> pd.Series:
        series = series.fillna(0)

        if formula == "linear":
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)

        elif formula == "inverse":
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return 1 - (series - min_val) / (max_val - min_val)

        elif formula == "binary":
            return series.notna().astype(float)

        elif formula == "step":
            def step_fn(x):
                if x <= 1:
                    return 0.3
                elif x == 2:
                    return 0.7
                else:
                    return 1.0

            return series.apply(step_fn)

        else:
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)