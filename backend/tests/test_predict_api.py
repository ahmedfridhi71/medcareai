"""
Tests for prediction API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPredictEndpoint:
    """Tests for /api/v1/predict endpoint."""

    @pytest.fixture
    def mock_prediction_result(self):
        """Standard prediction result for mocking."""
        return {
            "primary_prediction": "Influenza",
            "confidence": 0.92,
            "icd10_code": "J11",
            "severity": "moderate",
            "top_predictions": [
                {
                    "disease": "Influenza",
                    "confidence": 0.92,
                    "icd10_code": "J11",
                    "severity": "moderate",
                },
                {
                    "disease": "Common Cold",
                    "confidence": 0.05,
                    "icd10_code": "J00",
                    "severity": "mild",
                },
            ],
            "input_symptoms": ["fever", "cough", "body_ache"],
        }

    @pytest.mark.anyio
    async def test_predict_endpoint_success(self, client, mock_prediction_result):
        """Test successful prediction request."""
        with patch("app.api.v1.predict.get_predictor") as mock_get_predictor:
            mock_predictor = MagicMock()
            mock_predictor.get_available_symptoms.return_value = [
                "fever", "cough", "body_ache", "fatigue"
            ]
            mock_predictor.predict.return_value = mock_prediction_result
            mock_get_predictor.return_value = mock_predictor

            response = await client.post(
                "/api/v1/predict/",
                json={"symptoms": ["fever", "cough", "body_ache"]},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["primary_prediction"] == "Influenza"
            assert data["confidence"] == 0.92

    @pytest.mark.anyio
    async def test_predict_endpoint_empty_symptoms(self, client):
        """Test prediction with empty symptoms returns error."""
        response = await client.post(
            "/api/v1/predict/",
            json={"symptoms": []},
        )

        # FastAPI validation error for min_length=1
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_symptoms_endpoint(self, client):
        """Test GET /symptoms endpoint."""
        with patch("app.api.v1.predict.get_predictor") as mock_get_predictor:
            mock_predictor = MagicMock()
            mock_predictor.get_available_symptoms.return_value = [
                "fever", "cough", "fatigue"
            ]
            mock_get_predictor.return_value = mock_predictor

            response = await client.get("/api/v1/predict/symptoms")

            assert response.status_code == 200
            data = response.json()
            assert "symptoms" in data
            assert "count" in data
            assert data["count"] == 3
