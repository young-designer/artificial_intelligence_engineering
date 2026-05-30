from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.config import get_settings
from src.train import main as train_main


settings = get_settings()
if not settings.model_path.exists() or not settings.model_metadata_path.exists():
    train_main()

from src.service import app  # noqa: E402


def valid_payload() -> dict:
    return {
        "orders": [
            {
                "delivery_type": "express",
                "region_type": "suburb",
                "payment_type": "prepaid",
                "package_size": "large",
                "traffic_level": "high",
                "weather_condition": "rain",
                "dispatch_shift": "day",
                "is_weekend": True,
                "is_holiday": False,
                "fragile_item": False,
                "requires_call_before_delivery": False,
                "order_value": 12990.0,
                "items_count": 3,
                "delivery_distance_km": 24.2,
                "promised_hours": 18.0,
                "minutes_since_order": 420,
                "warehouse_backlog": 11,
                "courier_experience_months": 12.0,
                "courier_load_today": 10,
                "historical_late_rate": 0.26,
                "previous_failed_deliveries": 1,
                "address_quality_score": 0.62,
                "route_complexity": 6.4,
                "customer_priority_score": 2,
            }
        ]
    }


class DeliveryRiskServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model_name", payload)

    def test_predict_returns_probability_and_action(self) -> None:
        response = self.client.post("/predict", json=valid_payload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["predictions"]), 1)
        prediction = body["predictions"][0]
        self.assertGreaterEqual(prediction["late_probability"], 0.0)
        self.assertLessEqual(prediction["late_probability"], 1.0)
        self.assertIn(prediction["risk_level"], ["low", "medium", "high"])
        self.assertTrue(prediction["recommended_action"])

    def test_invalid_payload_is_rejected(self) -> None:
        payload = valid_payload()
        payload["orders"][0]["delivery_distance_km"] = -1
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("delivery_requests_total", response.text)


if __name__ == "__main__":
    unittest.main()
