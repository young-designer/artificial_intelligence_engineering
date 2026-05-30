from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings:
    def __init__(self) -> None:
        _load_env_file(PROJECT_ROOT / ".env")
        self.app_name = os.getenv("APP_NAME", "delivery-delay-risk-service")
        self.app_env = os.getenv("APP_ENV", "development")
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.model_path = _project_path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
        self.model_metadata_path = _project_path(
            os.getenv("MODEL_METADATA_PATH", "artifacts/model_metadata.json")
        )
        self.config_path = _project_path(os.getenv("CONFIG_PATH", "configs/train_config.yaml"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_train_config(config_path: Path | None = None) -> dict[str, Any]:
    target = config_path or get_settings().config_path
    with target.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
