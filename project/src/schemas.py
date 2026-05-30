from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DeliveryOrderFeatures(BaseModel):
    delivery_type: Literal["standard", "express", "same_day", "scheduled"]
    region_type: Literal["city_center", "residential", "suburb", "industrial", "remote_area"]
    payment_type: Literal["prepaid", "card_on_delivery", "cash_on_delivery"]
    package_size: Literal["small", "medium", "large", "bulky"]
    traffic_level: Literal["low", "normal", "high", "severe"]
    weather_condition: Literal["clear", "rain", "snow", "storm"]
    dispatch_shift: Literal["morning", "day", "evening", "night"]
    is_weekend: bool
    is_holiday: bool
    fragile_item: bool
    requires_call_before_delivery: bool
    order_value: float = Field(ge=0.0, le=100000.0)
    items_count: int = Field(ge=1, le=50)
    delivery_distance_km: float = Field(gt=0.0, le=150.0)
    promised_hours: float = Field(ge=1.0, le=120.0)
    minutes_since_order: int = Field(ge=0, le=7200)
    warehouse_backlog: int = Field(ge=0, le=100)
    courier_experience_months: float = Field(ge=0.0, le=240.0)
    courier_load_today: int = Field(ge=1, le=60)
    historical_late_rate: float = Field(ge=0.0, le=1.0)
    previous_failed_deliveries: int = Field(ge=0, le=20)
    address_quality_score: float = Field(ge=0.0, le=1.0)
    route_complexity: float = Field(ge=1.0, le=10.0)
    customer_priority_score: int = Field(ge=1, le=4)


class PredictionRequest(BaseModel):
    orders: list[DeliveryOrderFeatures] = Field(min_length=1, max_length=100)


class PredictionItem(BaseModel):
    is_late_delivery: int
    late_probability: float
    risk_level: Literal["low", "medium", "high"]
    recommended_action: str


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[PredictionItem]
