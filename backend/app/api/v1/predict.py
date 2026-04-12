"""
Prediction API Router.

Handles disease prediction, explanation, and symptom listing endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.prediction import (
    ErrorResponse,
    ExplanationResponse,
    PredictionRequest,
    PredictionResponse,
    SymptomsListResponse,
)
from app.services.ml_service import (
    MLServiceError,
    explain_prediction,
    get_available_symptoms,
    predict_disease,
)
from ml.predictor import DiseasePredictor, get_predictor

router = APIRouter()


async def get_ml_predictor() -> DiseasePredictor:
    """Dependency to get the ML predictor."""
    try:
        return get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": "ML model not available. Please train the model first.",
                "code": "MODEL_NOT_FOUND",
            },
        )


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid symptoms"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    summary="Predict disease from symptoms",
    description="Takes a list of symptoms and returns disease predictions with confidence scores.",
)
async def predict(
    request: PredictionRequest,
    predictor: Annotated[DiseasePredictor, Depends(get_ml_predictor)],
) -> PredictionResponse:
    """
    Predict disease from symptoms.

    Returns top-k predictions with confidence scores and disease metadata.
    """
    try:
        result = await predict_disease(
            predictor=predictor,
            symptoms=request.symptoms,
            top_k=request.top_k,
        )
        return PredictionResponse(**result)
    except MLServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": e.message, "code": e.code},
        )


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid symptoms"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
    summary="Explain prediction",
    description="Returns SHAP-based explanation showing which symptoms contributed to the prediction.",
)
async def explain(
    request: PredictionRequest,
    predictor: Annotated[DiseasePredictor, Depends(get_ml_predictor)],
) -> ExplanationResponse:
    """
    Get SHAP explanation for a disease prediction.

    Shows which symptoms increased or decreased the prediction probability.
    """
    try:
        result = await explain_prediction(
            predictor=predictor,
            symptoms=request.symptoms,
        )
        return ExplanationResponse(**result)
    except MLServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": e.message, "code": e.code},
        )


@router.get(
    "/symptoms",
    response_model=SymptomsListResponse,
    summary="List available symptoms",
    description="Returns all valid symptom names that can be used for predictions.",
)
async def list_symptoms(
    predictor: Annotated[DiseasePredictor, Depends(get_ml_predictor)],
) -> SymptomsListResponse:
    """
    Get list of all available symptoms.

    Use these symptom names when making prediction requests.
    """
    symptoms = await get_available_symptoms(predictor)
    return SymptomsListResponse(symptoms=symptoms, count=len(symptoms))
