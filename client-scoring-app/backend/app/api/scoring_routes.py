import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.manual_score_schema import ManualScoreRequest
from app.services.excel_service import ExcelService
from app.services.feature_config_service import FeatureConfigService
from app.services.response_service import ResponseService
from app.services.score_service import ScoreService
from scoring import pipeline


router = APIRouter(tags=["scoring"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def _delete_temp_file(
    file: UploadFile,
    temp_path: Path | None,
) -> None:
    """
    Закрывает UploadFile и удаляет временный файл.
    Несколько попыток нужны на Windows, где файл может кратко оставаться занят.
    """
    await file.close()

    if temp_path is None or not temp_path.exists():
        return

    for attempt in range(5):
        try:
            temp_path.unlink()
            return
        except PermissionError:
            if attempt == 4:
                print(
                    f"Не удалось удалить временный файл: {temp_path}"
                )
                return
            await asyncio.sleep(0.2)


@router.post("/score")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Query(...),
):
    temp_path: Path | None = None

    try:
        if not file.filename:
            raise HTTPException(400, "Имя файла отсутствует")

        if not file.filename.lower().endswith((".xlsx", ".xls")):
            raise HTTPException(
                400,
                "Неверный формат. Ожидается .xlsx или .xls",
            )

        if mode not in {"churn", "winback"}:
            raise HTTPException(
                400,
                "Параметр mode должен быть churn или winback",
            )

        file_id = str(uuid.uuid4())[:8]
        safe_filename = Path(file.filename).name
        temp_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

        content = await file.read()

        if not content:
            raise HTTPException(400, "Загруженный файл пуст")

        temp_path.write_bytes(content)

        is_valid, errors, df = ExcelService.validate_file(
            str(temp_path),
            mode,
        )

        if not is_valid:
            raise HTTPException(
                400,
                f"Ошибка валидации: {', '.join(errors)}",
            )

        # Здесь действительно вызывается CatBoost pipeline.
        result_df = pipeline.calculate_scores(df)

        # ResponseService теперь понимает имена Final_Probability,
        # Risk_Level, Feature_Completeness и Top_Factors.
        response_data = ResponseService.prepare_response(result_df)
        response_data["file_id"] = file_id
        response_data["message"] = "Файл успешно обработан"
        response_data["status"] = "success"

        ResponseService.save_result(response_data, file_id)

        return JSONResponse(content=response_data)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            500,
            f"Внутренняя ошибка: {exc}",
        ) from exc

    finally:
        await _delete_temp_file(file, temp_path)


@router.post("/score/manual")
async def manual_score(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    temp_path: Path | None = None

    try:
        if not file.filename:
            raise HTTPException(400, "Имя файла отсутствует")

        if not file.filename.lower().endswith((".xlsx", ".xls")):
            raise HTTPException(
                400,
                "Неверный формат. Ожидается .xlsx или .xls",
            )

        if config:
            try:
                request_data = json.loads(config)
                request = ManualScoreRequest(**request_data)
                feature_list = request.features

            except json.JSONDecodeError as exc:
                raise HTTPException(
                    400,
                    "Неверный JSON в поле config",
                ) from exc

            except Exception as exc:
                raise HTTPException(
                    400,
                    f"Ошибка валидации признаков: {exc}",
                ) from exc

        else:
            feature_list = FeatureConfigService.list_features()

            if not feature_list:
                raise HTTPException(
                    400,
                    "Признаки не переданы и не настроены через /features",
                )

        risk_total_weight = round(
            sum(
                float(feature.risk_weight)
                for feature in feature_list
            ),
            3,
        )
        value_total_weight = round(
            sum(
                float(feature.value_weight)
                for feature in feature_list
            ),
            3,
        )

        if round(risk_total_weight, 3) != 1.000:
            raise HTTPException(
                400,
                "Сумма весов риска должна быть равна 1.000. "
                f"Текущая сумма: {risk_total_weight:.3f}",
            )

        if round(value_total_weight, 3) != 1.000:
            raise HTTPException(
                400,
                "Сумма весов ценности должна быть равна 1.000. "
                f"Текущая сумма: {value_total_weight:.3f}",
            )

        file_id = str(uuid.uuid4())[:8]
        safe_filename = Path(file.filename).name
        temp_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

        content = await file.read()

        if not content:
            raise HTTPException(400, "Загруженный файл пуст")

        temp_path.write_bytes(content)

        is_valid, errors, df = ExcelService.validate_manual_file(
            str(temp_path),
            feature_list,
        )

        if not is_valid:
            raise HTTPException(
                400,
                f"Ошибка валидации файла: {', '.join(errors)}",
            )

        result_df = ScoreService.calculate_manual_score(
            df,
            feature_list,
        )

        selected_feature_names = [
            str(feature.name).strip()
            for feature in feature_list
        ]

        # Передаём текущий список признаков.
        # Добавленные попадут в JSON/Excel, удалённые — нет.
        response_data = ResponseService.prepare_response(
            result_df,
            extra_columns=selected_feature_names,
        )
        response_data["file_id"] = file_id
        response_data["message"] = "Ручной скоринг выполнен успешно"
        response_data["status"] = "success"
        response_data["selected_features"] = selected_feature_names
        response_data["source_sheet"] = df.attrs.get("source_sheet")

        ResponseService.save_result(response_data, file_id)

        return JSONResponse(content=response_data)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            500,
            f"Внутренняя ошибка: {exc}",
        ) from exc

    finally:
        await _delete_temp_file(file, temp_path)


@router.get("/download/{file_id}")
async def download_result(file_id: str):
    result = ResponseService.get_result(file_id)

    if result is None:
        raise HTTPException(
            404,
            f"Результат {file_id} не найден",
        )

    return JSONResponse(content=result)
