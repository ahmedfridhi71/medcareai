"""
Tests for ML prediction functionality.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Test data paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "ml" / "artifacts"


class TestDataset:
    """Tests for the diseases-symptoms dataset."""

    def test_dataset_file_exists(self):
        """Test that the dataset CSV file exists."""
        dataset_file = ARTIFACTS_DIR / "diseases_symptoms.csv"
        assert dataset_file.exists(), "diseases_symptoms.csv should exist"


class TestMLService:
    """Tests for ML service functions."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor for testing."""
        predictor = MagicMock()
        predictor.get_available_symptoms.return_value = [
            "fever", "cough", "fatigue", "headache"
        ]
        predictor.predict.return_value = {
            "primary_prediction": "Common Cold",
            "confidence": 0.85,
            "icd10_code": "J00",
            "severity": "mild",
            "top_predictions": [
                {
                    "disease": "Common Cold",
                    "confidence": 0.85,
                    "icd10_code": "J00",
                    "severity": "mild",
                },
            ],
            "input_symptoms": ["fever", "cough"],
        }
        return predictor

    @pytest.mark.anyio
    async def test_predict_disease_success(self, mock_predictor):
        """Test successful disease prediction."""
        from app.services.ml_service import predict_disease

        result = await predict_disease(
            predictor=mock_predictor,
            symptoms=["fever", "cough"],
            top_k=3,
        )

        assert result["primary_prediction"] == "Common Cold"
        assert result["confidence"] == 0.85
        mock_predictor.predict.assert_called_once()

    @pytest.mark.anyio
    async def test_predict_disease_empty_symptoms(self, mock_predictor):
        """Test that empty symptoms raises error."""
        from app.services.ml_service import MLServiceError, predict_disease

        with pytest.raises(MLServiceError) as exc:
            await predict_disease(
                predictor=mock_predictor,
                symptoms=[],
                top_k=3,
            )

        assert exc.value.code == "EMPTY_SYMPTOMS"

    @pytest.mark.anyio
    async def test_predict_disease_invalid_symptoms(self, mock_predictor):
        """Test that invalid symptoms raises error."""
        from app.services.ml_service import MLServiceError, predict_disease

        with pytest.raises(MLServiceError) as exc:
            await predict_disease(
                predictor=mock_predictor,
                symptoms=["invalid_symptom"],
                top_k=3,
            )

        assert exc.value.code == "INVALID_SYMPTOMS"


class TestPredictionSchemas:
    """Tests for prediction Pydantic schemas."""

    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        from app.schemas.prediction import PredictionRequest

        request = PredictionRequest(
            symptoms=["fever", "cough"],
            top_k=3,
        )

        assert request.symptoms == ["fever", "cough"]
        assert request.top_k == 3

    def test_prediction_request_defaults(self):
        """Test prediction request default values."""
        from app.schemas.prediction import PredictionRequest

        request = PredictionRequest(symptoms=["fever"])

        assert request.top_k == 3  # default value

    def test_prediction_response_valid(self):
        """Test valid prediction response."""
        from app.schemas.prediction import PredictionResponse

        response = PredictionResponse(
            primary_prediction="Common Cold",
            confidence=0.85,
            icd10_code="J00",
            severity="mild",
            top_predictions=[
                {
                    "disease": "Common Cold",
                    "confidence": 0.85,
                    "icd10_code": "J00",
                    "severity": "mild",
                }
            ],
            input_symptoms=["fever", "cough"],
        )

        assert response.primary_prediction == "Common Cold"
        assert response.confidence == 0.85
