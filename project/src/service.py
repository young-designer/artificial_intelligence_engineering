from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from src.config import get_settings
from src.features import make_feature_frame
from src.logging_utils import configure_logging
from src.schemas import PredictionItem, PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)


class ServiceMetrics:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests_total = 0
        self.predicted_orders_total = 0
        self.errors_total = 0
        self.last_latency_ms = 0.0

    def to_prometheus(self) -> str:
        uptime = time.time() - self.started_at
        lines = [
            "# HELP delivery_requests_total Total HTTP requests processed by the service",
            "# TYPE delivery_requests_total counter",
            f"delivery_requests_total {self.requests_total}",
            "# HELP delivery_predicted_orders_total Total orders scored by the model",
            "# TYPE delivery_predicted_orders_total counter",
            f"delivery_predicted_orders_total {self.predicted_orders_total}",
            "# HELP delivery_errors_total Total failed requests or prediction errors",
            "# TYPE delivery_errors_total counter",
            f"delivery_errors_total {self.errors_total}",
            "# HELP delivery_last_request_latency_ms Latency of the last request in milliseconds",
            "# TYPE delivery_last_request_latency_ms gauge",
            f"delivery_last_request_latency_ms {self.last_latency_ms:.2f}",
            "# HELP delivery_service_uptime_seconds Uptime of the API process",
            "# TYPE delivery_service_uptime_seconds gauge",
            f"delivery_service_uptime_seconds {uptime:.2f}",
        ]
        return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def risk_level(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["high"]:
        return "high"
    if probability >= thresholds["medium"]:
        return "medium"
    return "low"


def action_for_risk(level: str) -> str:
    actions = {
        "high": "Переназначить курьера или предупредить клиента о риске задержки.",
        "medium": "Проверить маршрут и держать заказ в приоритетном мониторинге.",
        "low": "Стандартная обработка без ручной эскалации.",
    }
    return actions[level]


settings = get_settings()
configure_logging(settings.log_level)
metrics = ServiceMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.model_path.exists() or not settings.model_metadata_path.exists():
        raise RuntimeError("Model artifacts are missing. Run `python -m src.train` first.")
    app.state.model = joblib.load(settings.model_path)
    app.state.metadata = load_json(settings.model_metadata_path)
    LOGGER.info("Loaded model %s from %s", app.state.metadata["best_model_name"], settings.model_path)
    yield


app = FastAPI(
    title="Delivery Delay Risk Service",
    version="1.0.0",
    description="API for estimating the risk of late order delivery.",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    started = time.perf_counter()
    metrics.requests_total += 1
    try:
        response = await call_next(request)
    except Exception:
        metrics.errors_total += 1
        LOGGER.exception("Unhandled error: %s %s", request.method, request.url.path)
        raise
    finally:
        metrics.last_latency_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code >= 400:
        metrics.errors_total += 1
    LOGGER.info(
        "%s %s -> %s in %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        metrics.last_latency_ms,
    )
    return response


@app.get("/")
def root() -> dict[str, Any]:
    model_name = getattr(app.state, "metadata", {}).get("best_model_name", "not_loaded")
    return {
        "service": settings.app_name,
        "status": "running",
        "model_name": model_name,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "model_name": app.state.metadata["best_model_name"],
        "artifact_rows": app.state.metadata["rows"],
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    metadata = app.state.metadata
    return {
        "best_model_name": metadata["best_model_name"],
        "best_metrics": metadata["best_metrics"],
        "selection_metric": metadata["selection_metric"],
        "feature_count": len(metadata["feature_columns"]),
        "class_balance": metadata["class_balance"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.to_prometheus()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        metadata = app.state.metadata
        frame = make_feature_frame(
            [order.model_dump() for order in payload.orders],
            feature_columns=metadata["feature_columns"],
            boolean_columns=metadata["boolean_features"],
        )
        probabilities = app.state.model.predict_proba(frame)[:, 1]
        threshold = float(metadata["decision_threshold"])
        thresholds = metadata["risk_thresholds"]
        predictions = []
        for probability in probabilities:
            value = float(probability)
            level = risk_level(value, thresholds)
            predictions.append(
                PredictionItem(
                    is_late_delivery=int(value >= threshold),
                    late_probability=round(value, 4),
                    risk_level=level,
                    recommended_action=action_for_risk(level),
                )
            )

        metrics.predicted_orders_total += len(predictions)
        return PredictionResponse(model_name=metadata["best_model_name"], predictions=predictions)
    except ValueError as exc:
        metrics.errors_total += 1
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        metrics.errors_total += 1
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
