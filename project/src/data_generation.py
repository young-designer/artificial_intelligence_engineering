from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGION_DISTANCE_BONUS = {
    "city_center": -1.5,
    "residential": 1.0,
    "suburb": 5.5,
    "industrial": 7.0,
    "remote_area": 15.0,
}

WEATHER_RISK = {
    "clear": 0.0,
    "rain": 0.35,
    "snow": 0.65,
    "storm": 1.0,
}

TRAFFIC_RISK = {
    "low": 0.0,
    "normal": 0.25,
    "high": 0.65,
    "severe": 1.0,
}

DELIVERY_TYPE_HOURS = {
    "same_day": 6,
    "express": 18,
    "standard": 48,
    "scheduled": 72,
}

DELIVERY_TYPE_RISK = {
    "same_day": 0.55,
    "express": 0.28,
    "standard": 0.0,
    "scheduled": -0.2,
}

PACKAGE_RISK = {
    "small": 0.0,
    "medium": 0.08,
    "large": 0.22,
    "bulky": 0.38,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_delivery_orders(rows: int = 6500, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    delivery_type = rng.choice(
        ["standard", "express", "same_day", "scheduled"],
        size=rows,
        p=[0.46, 0.28, 0.16, 0.10],
    )
    region_type = rng.choice(
        ["city_center", "residential", "suburb", "industrial", "remote_area"],
        size=rows,
        p=[0.20, 0.36, 0.22, 0.14, 0.08],
    )
    payment_type = rng.choice(
        ["prepaid", "card_on_delivery", "cash_on_delivery"],
        size=rows,
        p=[0.58, 0.27, 0.15],
    )
    package_size = rng.choice(
        ["small", "medium", "large", "bulky"],
        size=rows,
        p=[0.38, 0.37, 0.18, 0.07],
    )
    traffic_level = rng.choice(
        ["low", "normal", "high", "severe"],
        size=rows,
        p=[0.18, 0.48, 0.25, 0.09],
    )
    weather_condition = rng.choice(
        ["clear", "rain", "snow", "storm"],
        size=rows,
        p=[0.60, 0.25, 0.10, 0.05],
    )
    dispatch_shift = rng.choice(
        ["morning", "day", "evening", "night"],
        size=rows,
        p=[0.28, 0.39, 0.24, 0.09],
    )

    is_weekend = rng.binomial(1, 0.29, rows).astype(bool)
    is_holiday = rng.binomial(1, 0.07, rows).astype(bool)
    fragile_item = rng.binomial(1, np.where(package_size == "bulky", 0.18, 0.09), rows).astype(bool)
    requires_call_before_delivery = (
        rng.binomial(1, np.where(payment_type == "cash_on_delivery", 0.70, 0.24), rows).astype(bool)
    )

    order_value = np.round(np.clip(rng.lognormal(mean=8.2, sigma=0.75, size=rows), 350, 65000), 2)
    items_count = np.clip(rng.poisson(2.4, rows) + 1, 1, 18)

    raw_distance = rng.gamma(shape=2.2, scale=4.2, size=rows)
    distance_bonus = np.array([REGION_DISTANCE_BONUS[value] for value in region_type])
    package_distance_noise = np.where(package_size == "bulky", rng.normal(2.0, 1.5, rows), 0.0)
    delivery_distance_km = np.round(
        np.clip(raw_distance + distance_bonus + package_distance_noise, 0.7, 95.0),
        2,
    )

    base_promised_hours = np.array([DELIVERY_TYPE_HOURS[value] for value in delivery_type])
    promised_hours = np.clip(base_promised_hours + rng.normal(0, 3.0, rows), 4, 96).round(1)
    minutes_since_order = np.clip(
        rng.normal(promised_hours * 18, promised_hours * 5, rows),
        15,
        promised_hours * 60 * 0.95,
    ).round(0).astype(int)

    warehouse_backlog = np.clip(
        rng.poisson(5.5, rows)
        + is_weekend.astype(int) * rng.integers(0, 4, rows)
        + is_holiday.astype(int) * rng.integers(2, 7, rows),
        0,
        28,
    )
    courier_experience_months = np.round(np.clip(rng.gamma(shape=2.0, scale=13.0, size=rows), 1, 144), 1)
    courier_load_today = np.clip(rng.poisson(7.5, rows) + (delivery_type == "same_day").astype(int) * 2, 1, 24)
    historical_late_rate = np.round(np.clip(rng.beta(2.2, 8.0, rows) + (region_type == "remote_area") * 0.10, 0, 0.82), 3)
    previous_failed_deliveries = np.clip(
        rng.poisson(0.25 + historical_late_rate * 2.4, rows),
        0,
        8,
    )
    address_quality_score = np.round(
        np.clip(rng.beta(7.0, 2.2, rows) - (region_type == "remote_area") * 0.12, 0.12, 1.0),
        3,
    )
    route_complexity = np.round(
        np.clip(
            delivery_distance_km / 9.0
            + np.array([TRAFFIC_RISK[value] for value in traffic_level]) * 2.2
            + np.array([WEATHER_RISK[value] for value in weather_condition]) * 1.6
            + rng.normal(1.2, 0.7, rows),
            1.0,
            10.0,
        ),
        2,
    )
    customer_priority_score = rng.choice([1, 2, 3, 4], size=rows, p=[0.32, 0.38, 0.22, 0.08])

    weather_risk = np.array([WEATHER_RISK[value] for value in weather_condition])
    traffic_risk = np.array([TRAFFIC_RISK[value] for value in traffic_level])
    delivery_risk = np.array([DELIVERY_TYPE_RISK[value] for value in delivery_type])
    package_risk = np.array([PACKAGE_RISK[value] for value in package_size])

    logit = (
        -2.65
        + 0.045 * delivery_distance_km
        + 0.125 * warehouse_backlog
        + 0.115 * courier_load_today
        + 0.34 * route_complexity
        + 1.18 * traffic_risk
        + 0.98 * weather_risk
        + delivery_risk
        + package_risk
        + 0.34 * is_weekend.astype(int)
        + 0.58 * is_holiday.astype(int)
        + 0.18 * fragile_item.astype(int)
        + 0.23 * requires_call_before_delivery.astype(int)
        + 0.24 * previous_failed_deliveries
        + 1.45 * historical_late_rate
        - 0.012 * courier_experience_months
        - 0.95 * address_quality_score
        - 0.055 * promised_hours
        - 0.08 * customer_priority_score
        + rng.normal(0, 0.7, rows)
    )
    late_probability = _sigmoid(logit)
    is_late_delivery = rng.binomial(1, late_probability).astype(int)

    risk_segment = np.select(
        [late_probability >= 0.68, late_probability >= 0.38],
        ["high", "medium"],
        default="low",
    )

    return pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + index}" for index in range(rows)],
            "delivery_type": delivery_type,
            "region_type": region_type,
            "payment_type": payment_type,
            "package_size": package_size,
            "traffic_level": traffic_level,
            "weather_condition": weather_condition,
            "dispatch_shift": dispatch_shift,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "fragile_item": fragile_item,
            "requires_call_before_delivery": requires_call_before_delivery,
            "order_value": order_value,
            "items_count": items_count,
            "delivery_distance_km": delivery_distance_km,
            "promised_hours": promised_hours,
            "minutes_since_order": minutes_since_order,
            "warehouse_backlog": warehouse_backlog,
            "courier_experience_months": courier_experience_months,
            "courier_load_today": courier_load_today,
            "historical_late_rate": historical_late_rate,
            "previous_failed_deliveries": previous_failed_deliveries,
            "address_quality_score": address_quality_score,
            "route_complexity": route_complexity,
            "customer_priority_score": customer_priority_score,
            "simulated_late_probability": np.round(late_probability, 4),
            "simulated_risk_segment": risk_segment,
            "is_late_delivery": is_late_delivery,
        }
    )


def save_dataset(path: Path, rows: int = 6500, random_seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = generate_delivery_orders(rows=rows, random_seed=random_seed)
    dataframe.to_csv(path, index=False)
    return path
