"""
MedCareAI ML Service.

Business logic layer for disease prediction functionality.
This service is framework-agnostic and does not import FastAPI.
"""

from typing import Dict, List

from ml.predictor import DiseasePredictor


class MLServiceError(Exception):
    """Custom exception for ML service errors."""

    def __init__(self, message: str, code: str = "ML_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


async def predict_disease(
    predictor: DiseasePredictor,
    symptoms: List[str],
    top_k: int = 3,
) -> Dict:
    """
    Predict disease from list of symptoms.

    Args:
        predictor: Loaded DiseasePredictor instance
        symptoms: List of symptom names
        top_k: Number of top predictions to return

    Returns:
        Prediction result dictionary

    Raises:
        MLServiceError: If symptoms are invalid or empty
    """
    if not symptoms:
        raise MLServiceError(
            "At least one symptom is required",
            code="EMPTY_SYMPTOMS",
        )

    # Validate symptoms
    valid_symptoms = set(predictor.get_available_symptoms())
    invalid_symptoms = [s for s in symptoms if s not in valid_symptoms]

    if invalid_symptoms:
        raise MLServiceError(
            f"Invalid symptoms: {', '.join(invalid_symptoms)}",
            code="INVALID_SYMPTOMS",
        )

    # Get prediction
    result = predictor.predict(symptoms, top_k=top_k)

    return result


async def explain_prediction(
    predictor: DiseasePredictor,
    symptoms: List[str],
    max_features: int = 10,
) -> Dict:
    """
    Get SHAP explanation for a prediction.

    Args:
        predictor: Loaded DiseasePredictor instance
        symptoms: List of symptom names
        max_features: Max features in explanation

    Returns:
        Explanation dictionary with feature contributions
    """
    if not symptoms:
        raise MLServiceError(
            "At least one symptom is required",
            code="EMPTY_SYMPTOMS",
        )

    explanation = predictor.explain(symptoms, max_features=max_features)

    return explanation


async def get_available_symptoms(predictor: DiseasePredictor) -> Dict:
    """
    Get list of all valid symptom names using predictor.

    Args:
        predictor: Loaded DiseasePredictor instance

    Returns:
        Dictionary with symptoms list
    """
    return {"symptoms": predictor.get_available_symptoms()}


def get_symptoms_vocabulary() -> List[str]:
    """
    Get list of all valid symptom names directly from file.
    
    Use this when predictor is not available.

    Returns:
        List of symptom names
    """
    import json
    from pathlib import Path
    
    artifacts_dir = Path(__file__).parent.parent.parent / "ml" / "artifacts"
    feature_names_path = artifacts_dir / "feature_names.json"
    
    if feature_names_path.exists():
        with open(feature_names_path) as f:
            return json.load(f)
    return []
