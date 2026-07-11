import uuid
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from app.services.excel_service import ExcelService
from app.services.score_service import ScoreService
from app.services.response_service import ResponseService
from app.schemas.scoring_schema import FileUploadResponse
from app.schemas.manual_score_schema import ManualScoreRequest

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/score", response_model=FileUploadResponse)
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

        result_df = result_df.rename(columns={
            "final_probability": "final_probability",
            "risk_level": "risk_level",
            "feature_completeness": "feature_completeness",
            "top_factors": "top_factors"
        })

        response_data = ResponseService.prepare_response(result_df)
        ResponseService.save_result(response_data, file_id)

        temp_path.unlink()

        return FileUploadResponse(
            file_id=file_id,
            message="Файл успешно обработан",
            status="success"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Внутренняя ошибка: {str(e)}")


@router.post("/manual-score", response_model=FileUploadResponse)
async def manual_score(
        file: UploadFile = File(...),
        features: str = Form(...)
):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(400, "Неверный формат. Ожидается .xlsx или .xls")

        try:
            request_data = json.loads(features)
            request = ManualScoreRequest(**request_data)
        except json.JSONDecodeError:
            raise HTTPException(400, "Неверный формат JSON в поле features")
        except Exception as e:
            raise HTTPException(400, f"Ошибка валидации признаков: {str(e)}")

        file_id = str(uuid.uuid4())[:8]
        temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        is_valid, errors, df = ExcelService.validate_manual_file(str(temp_path))

        if not is_valid:
            temp_path.unlink()
            raise HTTPException(400, f"Ошибка валидации файла: {', '.join(errors)}")

        missing_columns = []
        for feature in request.features:
            if feature.name not in df.columns:
                missing_columns.append(feature.name)

        if missing_columns:
            temp_path.unlink()
            raise HTTPException(400, f"Отсутствуют колонки в файле: {', '.join(missing_columns)}")

        result_df = ScoreService.calculate_manual_score(df, request.features)
        response_data = ResponseService.prepare_response(result_df)
        ResponseService.save_result(response_data, file_id)

        temp_path.unlink()

        return FileUploadResponse(
            file_id=file_id,
            message="Ручной скоринг выполнен успешно",
            status="success"
        )

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