from fastapi import APIRouter, HTTPException

from app.schemas.manual_score_schema import (
    FeatureConfig,
    FeatureListResponse,
    FeatureDeleteResponse,
)
from app.services.feature_config_service import FeatureConfigService

router = APIRouter(prefix="/features", tags=["features"])


def _build_list_response() -> FeatureListResponse:
    features = FeatureConfigService.list_features()
    total_weight = round(sum(f.weight for f in features), 4)
    is_valid = bool(features) and 0.99 <= total_weight <= 1.01
    return FeatureListResponse(features=features, total_weight=total_weight, is_valid=is_valid)


@router.get("", response_model=FeatureListResponse)
async def list_features():
    """Текущий список признаков для ручного скоринга + сумма весов."""
    return _build_list_response()


@router.post("", response_model=FeatureListResponse)
async def add_feature(feature: FeatureConfig):
    """Добавить признак в конфиг. Сумма весов не обязана быть равна 1
    в момент добавления - это проверяется только при запуске расчёта
    (см. GET /features -> is_valid и POST /manual-score)."""
    try:
        FeatureConfigService.add_feature(feature)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _build_list_response()


@router.put("/{name}", response_model=FeatureListResponse)
async def update_feature(name: str, feature: FeatureConfig):
    """Обновить существующий признак (вес/тип/формулу)."""
    try:
        FeatureConfigService.update_feature(name, feature)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return _build_list_response()


@router.delete("/{name}", response_model=FeatureDeleteResponse)
async def delete_feature(name: str):
    """Удалить признак из конфига."""
    try:
        FeatureConfigService.delete_feature(name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return FeatureDeleteResponse(name=name, message="Признак удалён", status="success")