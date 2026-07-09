import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.excel_service import ExcelService
from app.services.response_service import ResponseService
from app.schemas.scoring_schema import FileUploadResponse

router = APIRouter()

# Временная папка для загруженных файлов
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/score", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(400, "Неверный формат. Ожидается .xlsx или .xls")

        file_id = str(uuid.uuid4())[:8]
        temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Валидация и расчёт
        is_valid, errors, df = ExcelService.validate_file(str(temp_path))

        if not is_valid:
            temp_path.unlink()
            raise HTTPException(400, f"Ошибка валидации: {', '.join(errors)}")

        result_df = ExcelService.calculate_scores(df)
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