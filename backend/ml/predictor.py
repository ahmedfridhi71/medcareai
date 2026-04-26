"""
MedCareAI Disease Predictor.

Handles model loading, inference, and SHAP explainability for disease predictions.
Designed to be loaded once at application startup and reused for all predictions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import shap


# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model.pkl"
LABEL_ENCODER_FILE = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURE_NAMES_FILE = ARTIFACTS_DIR / "feature_names.json"
DISEASE_INFO_FILE = ARTIFACTS_DIR / "disease_info.json"


class DiseasePredictor:
    """
    Disease prediction service using pre-trained ML model.

    Provides predictions with confidence scores and SHAP-based explanations.
    """

    def __init__(self):
        """Initialize predictor with model and metadata."""
        self.model = None
        self.label_encoder = None
        self.feature_names: List[str] = []
        self.disease_metadata: Dict = {}
        self.explainer = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load model artifacts from disk.

        Raises:
            FileNotFoundError: If model files don't exist
        """
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_FILE}. Run train.py first."
            )

        self.model = joblib.load(MODEL_FILE)
        self.label_encoder = joblib.load(LABEL_ENCODER_FILE)

        with open(FEATURE_NAMES_FILE, "r") as f:
            self.feature_names = json.load(f)

        with open(DISEASE_INFO_FILE, "r") as f:
            self.disease_metadata = json.load(f)

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def get_available_symptoms(self) -> List[str]:
        """Return list of valid symptom names."""
        return self.feature_names.copy()

    def symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert symptom names to binary feature vector.

        Args:
            symptoms: List of symptom names present in patient

        Returns:
            Binary numpy array of shape (1, n_features)
        """
        vector = np.zeros((1, len(self.feature_names)))
        for symptom in symptoms:
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                vector[0, idx] = 1
        return vector

    def predict(
        self,
        symptoms: List[str],
        top_k: int = 3,
    ) -> Dict:
        """
        Predict disease from symptoms.

        Args:
            symptoms: List of symptom names
            top_k: Number of top predictions to return

        Returns:
            Dictionary containing:
            - primary_prediction: Most likely disease
            - confidence: Confidence score (0-1)
            - top_predictions: List of top_k predictions with scores
            - icd10_code: ICD-10 code for primary prediction
            - severity: Severity level of primary prediction
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert symptoms to feature vector
        X = self.symptoms_to_vector(symptoms)

        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []

        for idx in top_indices:
            disease_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            metadata = self.disease_metadata.get(disease_name, {})

            top_predictions.append({
                "disease": disease_name,
                "confidence": round(confidence, 4),
                "icd10_code": metadata.get("icd10", "Unknown"),
                "severity": metadata.get("severity", "unknown"),
            })

        primary = top_predictions[0]

        return {
            "primary_prediction": primary["disease"],
            "confidence": primary["confidence"],
            "icd10_code": primary["icd10_code"],
            "severity": primary["severity"],
            "top_predictions": top_predictions,
            "input_symptoms": symptoms,
        }

    def explain(
        self,
        symptoms: List[str],
        max_features: int = 10,
    ) -> Dict:
        """
        Generate SHAP explanation for prediction.

        Args:
            symptoms: List of symptom names
            max_features: Maximum features to include in explanation

        Returns:
            Dictionary with feature contributions
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        X = self.symptoms_to_vector(symptoms)

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        # Get predicted class
        predicted_class = self.model.predict(X)[0]

        # Handle multi-class SHAP values
        # Newer SHAP returns ndarray of shape (n_samples, n_features, n_classes)
        # Older versions return list[ndarray] (one per class)
        import numpy as np
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class][0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # shape (n_samples, n_features, n_classes)
                class_shap_values = arr[0, :, predicted_class]
            elif arr.ndim == 2:
                class_shap_values = arr[0]
            else:
                class_shap_values = arr

        # Create feature importance dictionary
        importance = {}
        for i, feature in enumerate(self.feature_names):
            val = class_shap_values[i]
            # If still array-like (e.g. shape (n_classes,)), pick predicted class
            if hasattr(val, "shape") and getattr(val, "ndim", 0) > 0:
                val = val[predicted_class] if val.size > predicted_class else val.item()
            importance[feature] = float(val)

        # Sort by absolute importance
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:max_features]

        # Separate positive and negative contributors
        positive_contributors = [
            {"symptom": k, "contribution": round(v, 4)}
            for k, v in sorted_importance
            if v > 0
        ]
        negative_contributors = [
            {"symptom": k, "contribution": round(v, 4)}
            for k, v in sorted_importance
            if v < 0
        ]

        predicted_disease = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            "disease": predicted_disease,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
            "base_value": float(self.explainer.expected_value[predicted_class])
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else float(self.explainer.expected_value),
        }


# Global predictor instance (lazy loaded)
_predictor: Optional[DiseasePredictor] = None


def get_predictor() -> DiseasePredictor:
    """
    Get or create the global predictor instance.

    Loads model on first call.
    """
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
        _predictor.load()
    return _predictor
