import uuid
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

from app.services.excel_service import ExcelService
from app.services.score_service import ScoreService
from app.services.response_service import ResponseService
from app.services.feature_config_service import FeatureConfigService
from app.schemas.manual_score_schema import ManualScoreRequest

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/score")
async def upload_file(file: UploadFile = File(...), mode: str = Query(...)):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(400, "Неверный формат. Ожидается .xlsx или .xls")

        file_id = str(uuid.uuid4())[:8]
        temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        is_valid, errors, df = ExcelService.validate_file(str(temp_path), mode)

        if not is_valid:
            temp_path.unlink()
            raise HTTPException(400, f"Ошибка валидации: {', '.join(errors)}")

        result_df = ScoreService.calculate_auto_scores(df, mode)

        response_data = ResponseService.prepare_response(result_df)
        response_data["file_id"] = file_id
        response_data["message"] = "Файл успешно обработан"
        response_data["status"] = "success"
        ResponseService.save_result(response_data, file_id)

        temp_path.unlink()

        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")


@router.post("/manual-score")
async def manual_score(
        file: UploadFile = File(...),
        features: Optional[str] = Form(None),
):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(400, "Неверный формат. Ожидается .xlsx или .xls")

        if features:
            try:
                request_data = json.loads(features)
                request = ManualScoreRequest(**request_data)
            except json.JSONDecodeError:
                raise HTTPException(400, "Неверный формат JSON в поле features")
            except Exception as e:
                raise HTTPException(400, f"Ошибка валидации признаков: {str(e)}")
            feature_list = request.features
        else:
            feature_list = FeatureConfigService.list_features()
            if not feature_list:
                raise HTTPException(
                    400,
                    "Признаки не переданы в запросе и не сконфигурированы через /features"
                )
            total_weight = sum(f.weight for f in feature_list)
            if not (0.99 <= total_weight <= 1.01):
                raise HTTPException(
                    400,
                    f"Сумма весов сконфигурированных признаков должна быть равна 1.0 "
                    f"(сейчас {total_weight:.4f}). Проверьте /features"
                )

        file_id = str(uuid.uuid4())[:8]
        temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        is_valid, errors, df = ExcelService.validate_manual_file(str(temp_path), feature_list)

        if not is_valid:
            temp_path.unlink()
            raise HTTPException(400, f"Ошибка валидации файла: {', '.join(errors)}")

        result_df = ScoreService.calculate_manual_score(df, feature_list)
        response_data = ResponseService.prepare_response(result_df)
        response_data["file_id"] = file_id
        response_data["message"] = "Ручной скоринг выполнен успешно"
        response_data["status"] = "success"
        ResponseService.save_result(response_data, file_id)

        temp_path.unlink()

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")


@router.get("/download/{file_id}")
async def download_result(file_id: str):
    try:
        result = ResponseService.get_result(file_id)
        if result is None:
            raise HTTPException(404, f"Результат {file_id} не найден")
        return JSONResponse(content=result, media_type="application/json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка: {str(e)}")