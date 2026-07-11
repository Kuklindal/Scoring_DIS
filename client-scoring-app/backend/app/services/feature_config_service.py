import json
from pathlib import Path
from typing import List, Dict, Any

from app.schemas.manual_score_schema import FeatureConfig

CONFIG_DIR = Path("data")
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "features_config.json"


class FeatureConfigService:
    """
    CRUD над списком признаков для ручного скоринга.
    Хранится в JSON-файле на диске (data/features_config.json),
    переживает перезапуск сервера.
    """

    @staticmethod
    def _read_raw() -> List[Dict[str, Any]]:
        if not CONFIG_FILE.exists():
            return []
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    @staticmethod
    def _write_raw(data: List[Dict[str, Any]]) -> None:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def list_features() -> List[FeatureConfig]:
        return [FeatureConfig(**item) for item in FeatureConfigService._read_raw()]

    @staticmethod
    def add_feature(feature: FeatureConfig) -> None:
        data = FeatureConfigService._read_raw()
        if any(item["name"] == feature.name for item in data):
            raise ValueError(f"Признак '{feature.name}' уже существует")
        data.append(feature.dict())
        FeatureConfigService._write_raw(data)

    @staticmethod
    def update_feature(name: str, feature: FeatureConfig) -> None:
        data = FeatureConfigService._read_raw()
        idx = next((i for i, item in enumerate(data) if item["name"] == name), None)
        if idx is None:
            raise ValueError(f"Признак '{name}' не найден")
        data[idx] = feature.dict()
        FeatureConfigService._write_raw(data)

    @staticmethod
    def delete_feature(name: str) -> None:
        data = FeatureConfigService._read_raw()
        new_data = [item for item in data if item["name"] != name]
        if len(new_data) == len(data):
            raise ValueError(f"Признак '{name}' не найден")
        FeatureConfigService._write_raw(new_data)

    @staticmethod
    def get_total_weight() -> float:
        return sum(item["weight"] for item in FeatureConfigService._read_raw())

    @staticmethod
    def is_valid_total_weight() -> bool:
        data = FeatureConfigService._read_raw()
        if not data:
            return False
        total = sum(item["weight"] for item in data)
        return 0.99 <= total <= 1.01