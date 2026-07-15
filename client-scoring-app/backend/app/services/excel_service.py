import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional

from app.services.data_cleaning_service import DataCleaningService


REQUIRED_COLUMNS_CHURN = [
    "Контрагент",
    "Код ТО",
    "Срок жизни от регистрации",
    "Количество ОС",
    "Сумма счета ИО",
    "%Скидки",
    "Вид деятельности",
    "Годовая выручка",
    "Численность сотрудников",
    "Конкурент",
    "Была угроза отключения",
    "Время всех сессий в мин",
]

REQUIRED_COLUMNS_WINBACK = [
    *REQUIRED_COLUMNS_CHURN,
    "Причина отключения",
]

WEIGHTS = {
    "life_span": 0.075,
    "number_os": 0.10,
    "summa_scheta_io": 0.15,
    "discount": 0.075,
    "annual_revenue": 0.10,
    "number_of_employees": 0.05,
    "total_session_minutes": 0.15,
    "competitor": 0.10,
    "has_threat": 0.05,
    "reason": 0.15,
}

REASON_MULTIPLIERS = {
    "Ликвидация / Клиент пропал": 0.5,
    "ДОЛГИ (инициатива РИЦ)": 0.5,
    "Нет денег для оплаты счетов по К+": 0.75,
    "Есть наш К+ на другом ЮЛ или ТО": 0.75,
    "К+ от другого РИЦ": 0.75,
    "Экономят, сокращают расходы": 1.0,
    "Нет потребности в К+/сервисе": 1.0,
    "Конкуренты (кроме других РИЦ)": 1.0,
    "Есть предоплата. Отключение на короткий период": 1.0,
    "Правило1 по комплектам": 1.0,
    "НАВЕС": 1.0,
    "Изменение комплекта": 1.0,
}

RISK_LEVELS = [
    (0.0, 0.3, "Низкий"),
    (0.3, 0.6, "Средний"),
    (0.6, 0.8, "Высокий"),
    (0.8, 1.0, "Критический"),
]

MIN_PARSEABLE_RATIO = 0.5


class ExcelService:

    @staticmethod
    def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Убирает случайные пробелы из названий колонок Excel.
        Например, 'Скор ' становится 'Скор'.
        """
        result = df.copy()
        result.columns = [
            str(column).strip()
            for column in result.columns
        ]
        return result

    @staticmethod
    def _find_sheet_with_features(
        file_path: str,
        features: List[Any],
    ):
        required_names = {
            str(feature.name).strip()
            for feature in features
        }

        inspected = []

        # Контекстный менеджер гарантированно закроет Excel-файл.
        with pd.ExcelFile(file_path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                preview = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    header=0,
                    nrows=0,
                )
                preview = ExcelService._normalize_column_names(preview)

                columns = set(preview.columns)
                missing = sorted(required_names - columns)
                inspected.append((sheet_name, missing))

                if not missing:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=0,
                    )
                    df = ExcelService._normalize_column_names(df)

                    # Возвращаем DataFrame, а после выхода из with
                    # файл автоматически закрывается.
                    return sheet_name, df

        details = "; ".join(
            f"{sheet}: нет {', '.join(missing)}"
            for sheet, missing in inspected
        )

        raise ValueError(
            "Не найден лист, содержащий все выбранные признаки. "
            f"Проверены листы: {details}"
        )

    @staticmethod
    def validate_file(file_path: str, mode: str):
        errors = []

        try:
            if mode == "churn":
                sheet_name = "КИС_дляСкорингаДИС_Сопр"
                required_columns = REQUIRED_COLUMNS_CHURN
            else:
                sheet_name = "КИС_дляСкорингаДИС_Откл"
                required_columns = REQUIRED_COLUMNS_WINBACK

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=0,
            )
            df = ExcelService._normalize_column_names(df)

        except Exception as exc:
            errors.append(f"Ошибка чтения файла: {exc}")
            return False, errors, None

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            errors.append(
                f"Отсутствуют столбцы: {', '.join(missing)}"
            )

        if df.empty:
            errors.append("Файл не содержит данных")

        return len(errors) == 0, errors, df

    @staticmethod
    def normalize_series(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        minimum = numeric.min()
        maximum = numeric.max()

        if maximum == minimum:
            return pd.Series(0.5, index=numeric.index)

        return (numeric - minimum) / (maximum - minimum)

    @staticmethod
    def invert_normalize(series: pd.Series) -> pd.Series:
        return 1 - ExcelService.normalize_series(series)

    @staticmethod
    def get_reason_weight(reason) -> float:
        if pd.isna(reason):
            return 0
        return REASON_MULTIPLIERS.get(reason, 1.0)

    @staticmethod
    def validate_manual_file(
        file_path: str,
        features: Optional[List[Any]] = None,
    ):
        errors = []
        df = None

        try:
            if features:
                sheet_name, df = ExcelService._find_sheet_with_features(
                    file_path,
                    features,
                )
                # Техническое поле полезно для отладки, но не выводится клиенту.
                df.attrs["source_sheet"] = sheet_name
            else:
                with pd.ExcelFile(file_path) as excel_file:
                    first_sheet = excel_file.sheet_names[0]

                    df = pd.read_excel(
                        excel_file,
                        sheet_name=first_sheet,
                        header=0,
                    )

                df = ExcelService._normalize_column_names(df)
                df.attrs["source_sheet"] = first_sheet

        except Exception as exc:
            errors.append(f"Ошибка чтения файла: {exc}")
            return False, errors, None

        if df.empty:
            errors.append("Файл не содержит данных")
            return False, errors, df

        if features:
            missing = [
                feature.name
                for feature in features
                if str(feature.name).strip() not in df.columns
            ]

            if missing:
                errors.append(
                    f"Отсутствуют столбцы: {', '.join(missing)}"
                )

            for feature in features:
                feature_name = str(feature.name).strip()

                if feature_name not in df.columns:
                    continue

                raw = df[feature_name]
                cleaned = DataCleaningService.clean_column(
                    raw,
                    feature.type,
                )
                ratio = DataCleaningService.parseable_ratio(
                    raw,
                    cleaned,
                )

                if ratio < MIN_PARSEABLE_RATIO:
                    errors.append(
                        f"Колонка '{feature_name}': только "
                        f"{ratio:.0%} значений удалось распознать "
                        f"как тип '{feature.type}'."
                    )

        return len(errors) == 0, errors, df

    @staticmethod
    def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        norm = {
            "life": ExcelService.normalize_series(
                result["Срок жизни от регистрации"]
            ),
            "os": ExcelService.normalize_series(
                result["Количество ОС"]
            ),
            "invoice": ExcelService.normalize_series(
                result["Сумма счета ИО"]
            ),
            "discount": ExcelService.invert_normalize(
                result["%Скидки"]
            ),
            "revenue": ExcelService.normalize_series(
                result["Годовая выручка"]
            ),
            "employees": ExcelService.normalize_series(
                result["Численность сотрудников"]
            ),
            "sessions": ExcelService.normalize_series(
                result["Время всех сессий в мин"]
            ),
            "competitor": DataCleaningService.clean_binary(
                result["Конкурент"]
            ),
            "threat": DataCleaningService.clean_binary(
                result["Была угроза отключения"]
            ),
        }

        if "Причина отключения" not in result.columns:
            result["Причина отключения"] = None

        reason_weight = result[
            "Причина отключения"
        ].apply(ExcelService.get_reason_weight)

        weighted = (
            norm["life"] * WEIGHTS["life_span"]
            + norm["os"] * WEIGHTS["number_os"]
            + norm["invoice"] * WEIGHTS["summa_scheta_io"]
            + norm["discount"] * WEIGHTS["discount"]
            + norm["revenue"] * WEIGHTS["annual_revenue"]
            + norm["employees"] * WEIGHTS["number_of_employees"]
            + norm["sessions"] * WEIGHTS["total_session_minutes"]
            + norm["competitor"] * WEIGHTS["competitor"]
            + norm["threat"] * WEIGHTS["has_threat"]
            + reason_weight * WEIGHTS["reason"]
        )

        final_probability = weighted.clip(0, 1)
        result["final_probability"] = final_probability

        def get_risk(probability):
            for low, high, level in RISK_LEVELS:
                if low <= probability < high:
                    return level
            return "Критический"

        result["risk_level"] = final_probability.apply(get_risk)

        completeness_columns = [
            "Срок жизни от регистрации",
            "Количество ОС",
            "Сумма счета ИО",
            "%Скидки",
            "Годовая выручка",
            "Численность сотрудников",
            "Время всех сессий в мин",
            "Причина отключения",
        ]
        result["feature_completeness"] = (
            result[completeness_columns]
            .notna()
            .sum(axis=1)
            / len(completeness_columns)
        )

        def get_top(row):
            factors = []

            if DataCleaningService.clean_binary(
                pd.Series([row.get("Конкурент")])
            ).iloc[0] == 1:
                factors.append("Есть конкурент")

            if DataCleaningService.clean_binary(
                pd.Series([row.get("Была угроза отключения")])
            ).iloc[0] == 1:
                factors.append("Была угроза")

            if pd.notna(row.get("Причина отключения")):
                factors.append(
                    f"Причина: {row['Причина отключения']}"
                )

            if row.get("%Скидки", 0) > 20:
                factors.append("Высокая скидка")

            if row.get("Время всех сессий в мин", 999999) < 100:
                factors.append("Низкая активность")

            if row.get("Срок жизни от регистрации", 999) < 2:
                factors.append("Короткий срок жизни")

            return factors[:3]

        result["top_factors"] = result.apply(get_top, axis=1)
        return result