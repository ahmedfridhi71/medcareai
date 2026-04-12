"""
Pydantic schemas for prediction endpoints.

Defines request and response models for disease prediction API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for disease prediction."""

    symptoms: List[str] = Field(
        ...,
        min_length=1,
        description="List of symptom names present in patient",
        examples=[["fever", "cough", "fatigue"]],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top predictions to return",
    )


class DiseasePrediction(BaseModel):
    """Single disease prediction with confidence."""

    disease: str = Field(..., description="Predicted disease name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    icd10_code: str = Field(..., description="ICD-10 classification code")
    severity: str = Field(..., description="Disease severity level")


class PredictionResponse(BaseModel):
    """Response schema for disease prediction."""

    primary_prediction: str = Field(..., description="Most likely disease")
    confidence: float = Field(..., ge=0, le=1, description="Primary prediction confidence")
    icd10_code: str = Field(..., description="ICD-10 code for primary prediction")
    severity: str = Field(..., description="Severity level")
    top_predictions: List[DiseasePrediction] = Field(
        ...,
        description="Top-k disease predictions with confidence scores",
    )
    input_symptoms: List[str] = Field(..., description="Input symptoms used for prediction")


class FeatureContribution(BaseModel):
    """SHAP feature contribution for explanation."""

    symptom: str = Field(..., description="Symptom name")
    contribution: float = Field(..., description="SHAP contribution value")


class ExplanationResponse(BaseModel):
    """Response schema for prediction explanation."""

    disease: str = Field(..., description="Predicted disease")
    positive_contributors: List[FeatureContribution] = Field(
        ...,
        description="Symptoms that increased prediction probability",
    )
    negative_contributors: List[FeatureContribution] = Field(
        ...,
        description="Symptoms that decreased prediction probability",
    )
    base_value: float = Field(..., description="SHAP base value")


class SymptomsListResponse(BaseModel):
    """Response schema for available symptoms list."""

    symptoms: List[str] = Field(..., description="List of valid symptom names")
    count: int = Field(..., description="Total number of symptoms")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code for programmatic handling")
