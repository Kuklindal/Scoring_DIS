import re

import numpy as np
import pandas as pd


YES_VALUES = {
    "да", "есть", "true", "yes", "истина", "y", "1",
}

NO_VALUES = {
    "нет", "false", "no", "ложь", "n", "0", "отсутствует",
    "не было", "нету",
}

TYPE_ALIASES = {
    "binary": "binary",
    "бинарный": "binary",
    "numeric": "numeric",
    "числовой": "numeric",
    "number": "numeric",
    "percent": "percent",
    "процентный": "percent",
    "percentage": "percent",
    # Добавлено для скоринга возвращаемости (winback):
    # "Причина отключения" и подобные текстовые поля не приводятся
    # к числу, интерпретацию берёт на себя FormulaService.
    "categorical": "categorical",
    "категориальный": "categorical",
}


class DataCleaningService:
    """
    Очистка колонок перед ручным скорингом.

    Поддерживаются как системные значения типов:
      binary / numeric / percent / categorical

    так и значения, которые отправляет Streamlit:
      Бинарный / Числовой / Процентный / Категориальный
    """

    @staticmethod
    def normalize_feature_type(feature_type: str) -> str:
        key = str(feature_type or "").strip().lower()
        return TYPE_ALIASES.get(key, key)

    @staticmethod
    def _to_clean_str(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def clean_percent(series: pd.Series) -> pd.Series:
        """
        Примеры:
          '37%'   -> 37.0
          '37,5%' -> 37.5
          0.37    -> 0.37
          пусто   -> NaN

        Масштаб не меняется автоматически: нормализация выполняется
        FormulaService, поэтому допустимы и 0.37, и 37.
        """

        def parse(value):
            if pd.isna(value):
                return np.nan

            if isinstance(value, (int, float, np.number)):
                return float(value)

            text = (
                str(value)
                .strip()
                .replace("\xa0", "")
                .replace(" ", "")
                .replace("%", "")
                .replace(",", ".")
            )
            text = re.sub(r"[^\d.\-]", "", text)

            if not text or text in {"-", ".", "-."}:
                return np.nan

            try:
                return float(text)
            except ValueError:
                return np.nan

        return series.apply(parse).astype(float)

    @staticmethod
    def clean_numeric(series: pd.Series) -> pd.Series:
        """
        Приводит числа, денежные значения и строки с разделителями к float.
        Нераспознаваемые значения становятся NaN.
        """

        def parse(value):
            if pd.isna(value):
                return np.nan

            if isinstance(value, (int, float, np.number)):
                return float(value)

            text = str(value).strip().replace("\xa0", "").replace(" ", "")
            text = re.sub(r"[^\d,.\-]", "", text)

            if not text or text in {"-", ".", ",", "-.", "-,"}:
                return np.nan

            # 1,234.56 -> 1234.56
            # 1.234,56 -> 1234.56
            if "," in text and "." in text:
                if text.rfind(",") > text.rfind("."):
                    text = text.replace(".", "").replace(",", ".")
                else:
                    text = text.replace(",", "")
            else:
                text = text.replace(",", ".")

            if text.count(".") > 1:
                whole, _, fraction = text.rpartition(".")
                text = whole.replace(".", "") + "." + fraction

            try:
                return float(text)
            except ValueError:
                return np.nan

        return series.apply(parse).astype(float)

    clean_money = clean_numeric

    @staticmethod
    def clean_binary(series: pd.Series) -> pd.Series:
        """
        Правило бинаризации:
          - пустая ячейка / NaN -> 0.0;
          - 'Нет', False, 0 и аналоги -> 0.0;
          - 'Да', True, 1 и аналоги -> 1.0;
          - любое другое непустое слово -> 1.0.

        Например, название конкурента 'Главбух' означает наличие
        конкурента и преобразуется в 1.0.
        """

        def parse(value):
            text = DataCleaningService._to_clean_str(value)

            if not text:
                return 0.0

            if text in NO_VALUES:
                return 0.0

            if text in YES_VALUES:
                return 1.0

            try:
                number = float(text.replace(",", "."))
                return 0.0 if number == 0 else 1.0
            except ValueError:
                return 1.0

        return series.apply(parse).astype(float)

    @staticmethod
    def clean_categorical(series: pd.Series) -> pd.Series:
        """
        Категориальный тип НЕ приводится к числу - значения
        (например, текст причины отключения) остаются строками.
        Дальнейшую интерпретацию (например, вес причины отключения)
        выполняет FormulaService по исходному тексту.
        """

        def parse(value):
            if pd.isna(value):
                return None

            text = str(value).strip()
            return text if text else None

        return series.apply(parse)

    @staticmethod
    def clean_column(
        series: pd.Series,
        feature_type: str,
    ) -> pd.Series:
        normalized_type = DataCleaningService.normalize_feature_type(
            feature_type
        )

        if normalized_type == "percent":
            return DataCleaningService.clean_percent(series)

        if normalized_type == "numeric":
            return DataCleaningService.clean_numeric(series)

        if normalized_type == "binary":
            return DataCleaningService.clean_binary(series)

        if normalized_type == "categorical":
            return DataCleaningService.clean_categorical(series)

        raise ValueError(
            f"Неизвестный тип признака: '{feature_type}'. "
            "Допустимые типы: binary/Бинарный, "
            "numeric/Числовой, percent/Процентный, "
            "categorical/Категориальный."
        )

    @staticmethod
    def parseable_ratio(
        raw: pd.Series,
        cleaned: pd.Series,
    ) -> float:
        """
        Доля непустых исходных значений, которые удалось распознать.

        Для binary/categorical любое непустое значение распознаётся,
        поэтому доля будет 1.
        """
        raw_filled = raw.notna() & (
            raw.astype(str).str.strip() != ""
        )
        total_filled = int(raw_filled.sum())

        if total_filled == 0:
            return 1.0

        still_valid = int(
            (raw_filled & cleaned.notna()).sum()
        )
        return still_valid / total_filled
