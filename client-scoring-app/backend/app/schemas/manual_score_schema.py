from pydantic import BaseModel, Field, validator
from typing import List, Literal


class FeatureConfig(BaseModel):
    name: str = Field(..., description="Название признака")
    weight: float = Field(..., ge=0, le=1, description="Вес признака (0-1)")
    type: Literal["numeric", "binary", "percent"] = Field(..., description="Тип признака")
    formula: Literal["linear", "inverse", "binary", "step", "none"] = Field(..., description="Формула нормализации")


class ManualScoreRequest(BaseModel):
    features: List[FeatureConfig] = Field(..., min_items=1)

    @validator('features')
    def weights_sum_to_one(cls, v):
        total = sum(f.weight for f in v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Сумма весов должна быть равна 1.0 (сейчас {total:.4f})")
        return v