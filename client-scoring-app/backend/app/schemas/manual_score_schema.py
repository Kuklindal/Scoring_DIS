from typing import Any, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


FeatureType = Literal[
    "Бинарный",
    "Числовой",
    "Процентный",
    "binary",
    "numeric",
    "percent",
]

FeatureGroup = Literal[
    "risk",
    "value",
    "both",
]

FormulaType = Literal[
    "binary",
    "min_max",
    "min_max_inverse",
    "step_os",
    "linear",
    "inverse",
    "step",
    "none",
]


class FeatureConfig(BaseModel):
    name: str = Field(..., min_length=1)
    type: FeatureType
    group: FeatureGroup = "risk"

    risk_weight: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )
    value_weight: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    # Одна общая формула для риска и ценности.
    formula: Optional[FormulaType] = None

    # Старое поле оставлено для совместимости
    # с предыдущими конфигурациями.
    weight: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Название признака не может быть пустым"
            )

        return value

    @model_validator(mode="before")
    @classmethod
    def migrate_old_format(
        cls,
        data: Any,
    ) -> Any:
        """
        Поддержка старого формата:

        {
            "weight": 0.2,
            "group": "risk"
        }

        Старый weight переносится в risk_weight,
        value_weight или оба поля — в зависимости от group.
        """
        if not isinstance(data, dict):
            return data

        values = data.copy()

        group = values.get("group", "risk")
        old_weight = values.get("weight")

        if old_weight is not None:
            if (
                group in {"risk", "both"}
                and "risk_weight" not in values
            ):
                values["risk_weight"] = old_weight

            if (
                group in {"value", "both"}
                and "value_weight" not in values
            ):
                values["value_weight"] = old_weight

        # Поддержка предыдущего промежуточного формата
        # с двумя формулами.
        if "formula" not in values:
            old_risk_formula = values.get("risk_formula")
            old_value_formula = values.get("value_formula")

            values["formula"] = (
                old_risk_formula
                or old_value_formula
            )

        return values

    @model_validator(mode="after")
    def apply_default_formula(self):
        """
        Назначает формулу по умолчанию,
        если пользователь её не передал.
        """
        if self.formula is None:
            normalized_type = str(self.type).strip().lower()

            if normalized_type in {
                "binary",
                "бинарный",
            }:
                self.formula = "binary"
            else:
                self.formula = "min_max"

        return self


class ManualScoreRequest(BaseModel):
    features: List[FeatureConfig] = Field(
        ...,
        min_length=1,
    )

    target_column: str = "final_probability"

    @model_validator(mode="after")
    def validate_configuration(self):
        risk_total = sum(
            feature.risk_weight
            for feature in self.features
        )

        value_total = sum(
            feature.value_weight
            for feature in self.features
        )

        if round(risk_total, 3) != 1.000:
            raise ValueError(
                "Сумма весов риска должна быть равна 1.000 "
                f"(сейчас {risk_total:.3f})"
            )

        if round(value_total, 3) != 1.000:
            raise ValueError(
                "Сумма весов ценности должна быть равна 1.000 "
                f"(сейчас {value_total:.3f})"
            )

        normalized_names = [
            feature.name.strip().lower()
            for feature in self.features
        ]

        if len(normalized_names) != len(
            set(normalized_names)
        ):
            raise ValueError(
                "Названия признаков не должны повторяться"
            )

        return self


class FeatureListResponse(BaseModel):
    features: List[FeatureConfig]

    risk_total_weight: float = 0.0
    value_total_weight: float = 0.0

    is_risk_valid: bool = False
    is_value_valid: bool = False


class FeatureDeleteResponse(BaseModel):
    name: str
    message: str
    status: str
