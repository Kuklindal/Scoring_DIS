from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict

FeatureType = Literal["numeric", "binary", "percent"]
FormulaType = Literal["linear", "inverse", "binary", "step", "none"]

# Дефолтная формула нормализации, если пользователь её не указал явно
DEFAULT_FORMULAS: Dict[str, FormulaType] = {
    "numeric": "linear",
    "percent": "inverse",   # по умолчанию считаем, что чем больше % (скидка) - тем хуже
    "binary": "binary",
}


class FeatureConfig(BaseModel):
    name: str = Field(..., description="Название признака (колонка в Excel)")
    weight: float = Field(..., ge=0, le=1, description="Вес признака (0-1)")
    type: FeatureType = Field(..., description="Тип признака")
    formula: Optional[FormulaType] = Field(
        None,
        description="Формула нормализации. Если не указана - берётся дефолтная для данного типа"
    )

    @validator("formula", always=True)
    def apply_default_formula(cls, v, values):
        if v is not None:
            return v
        feature_type = values.get("type")
        return DEFAULT_FORMULAS.get(feature_type, "linear")


class ManualScoreRequest(BaseModel):
    features: List[FeatureConfig] = Field(..., min_items=1)

    @validator("features")
    def weights_sum_to_one(cls, v):
        total = sum(f.weight for f in v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Сумма весов должна быть равна 1.0 (сейчас {total:.4f})")
        return v


class FeatureListResponse(BaseModel):
    features: List[FeatureConfig]
    total_weight: float
    is_valid: bool


class FeatureDeleteResponse(BaseModel):
    name: str
    message: str
    status: str