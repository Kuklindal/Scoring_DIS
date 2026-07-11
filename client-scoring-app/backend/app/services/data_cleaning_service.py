import re
import numpy as np
import pandas as pd

YES_VALUES = {"да", "есть", "true", "yes", "истина", "y", "1"}
NO_VALUES = {"нет", "false", "no", "ложь", "n", "0", "отсутствует"}


class DataCleaningService:
    """
    Приведение произвольных колонок Excel к числовому виду для ручного
    скоринга. Работает по типу признака, который задаёт пользователь
    (numeric / percent / binary), а не по фиксированному набору колонок,
    как в основном (авто) скоринге.
    """

    @staticmethod
    def _to_clean_str(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def clean_percent(series: pd.Series) -> pd.Series:
        """'37%', '37,5%', 37, '37' -> float. Пустые/битые значения -> NaN."""

        def parse(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().replace("%", "").replace(",", ".")
            s = re.sub(r"[^\d.\-]", "", s)
            if not s or s in ("-", "."):
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan

        return series.apply(parse)

    @staticmethod
    def clean_money(series: pd.Series) -> pd.Series:
        """'44 794,58 ₽', '1,234.56', 44794.58 -> float. Пустые/битые -> NaN."""

        def parse(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().replace("\xa0", "").replace(" ", "")
            s = re.sub(r"[^\d,.\-]", "", s)
            if not s:
                return np.nan
            # если и запятая, и точка - запятая это разделитель тысяч
            if "," in s and "." in s:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
            if s.count(".") > 1:
                whole, _, frac = s.rpartition(".")
                s = whole.replace(".", "") + "." + frac
            try:
                return float(s)
            except ValueError:
                return np.nan

        return series.apply(parse)

    @staticmethod
    def clean_binary(series: pd.Series) -> pd.Series:
        """
        Да/Нет, True/False, 1/0 -> 1.0/0.0.
        Пустая ячейка -> NaN (нет данных).
        Любое другое непустое значение (например, название конкурента в
        колонке 'Конкурент') трактуется как "признак присутствует" -> 1.0.
        """

        def parse(v):
            s = DataCleaningService._to_clean_str(v)
            if not s:
                return np.nan
            if s in YES_VALUES:
                return 1.0
            if s in NO_VALUES:
                return 0.0
            try:
                num = float(s.replace(",", "."))
                return 1.0 if num != 0 else 0.0
            except ValueError:
                return 1.0  # непустое произвольное значение = признак присутствует

        return series.apply(parse)

    @staticmethod
    def clean_column(series: pd.Series, feature_type: str) -> pd.Series:
        if feature_type == "percent":
            return DataCleaningService.clean_percent(series)
        if feature_type == "numeric":
            return DataCleaningService.clean_money(series)
        if feature_type == "binary":
            return DataCleaningService.clean_binary(series)
        return series

    @staticmethod
    def parseable_ratio(raw: pd.Series, cleaned: pd.Series) -> float:
        """Доля строк, где исходное значение было непустым, но после очистки
        по заданному типу превратилось в NaN - индикатор несовпадения типа."""
        raw_filled = raw.notna() & (raw.astype(str).str.strip() != "")
        total_filled = int(raw_filled.sum())
        if total_filled == 0:
            return 1.0
        still_valid = int((raw_filled & cleaned.notna()).sum())
        return still_valid / total_filled