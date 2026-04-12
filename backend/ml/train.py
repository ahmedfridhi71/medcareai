"""
MedCareAI ML Model Training Script.

Loads the Kaggle diseases-symptoms dataset, trains a disease classification
model using RandomForest and XGBoost, and logs experiments to MLflow.
Dataset: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATASET_FILE = ARTIFACTS_DIR / "diseases_symptoms.csv"
MODEL_FILE = ARTIFACTS_DIR / "model.pkl"
LABEL_ENCODER_FILE = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURE_NAMES_FILE = ARTIFACTS_DIR / "feature_names.json"
DISEASE_INFO_FILE = ARTIFACTS_DIR / "disease_info.json"


def load_dataset() -> pd.DataFrame:
    """
    Load the diseases-symptoms CSV dataset.

    Returns:
        DataFrame with disease labels and symptom features
    """
    print(f"    Loading dataset from: {DATASET_FILE}")
    df = pd.read_csv(DATASET_FILE)
    return df


def prepare_features_and_labels(
    df: pd.DataFrame,
    sample_size: int = None,
    min_samples_per_class: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Extract features and labels from dataframe.

    Args:
        df: Raw dataframe with 'diseases' column and symptom columns
        sample_size: Optional limit on number of samples (for faster training)
        min_samples_per_class: Minimum samples required per disease class

    Returns:
        Tuple of (X features, y labels, feature names, label encoder)
    """
    # Filter out rare disease classes
    disease_counts = df.iloc[:, 0].value_counts()
    valid_diseases = disease_counts[disease_counts >= min_samples_per_class].index
    df = df[df.iloc[:, 0].isin(valid_diseases)]

    # Sample if requested (for faster development iterations)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # First column is the disease label
    diseases = df.iloc[:, 0].values
    feature_names = df.columns[1:].tolist()

    # Extract features (all columns except first)
    X = df.iloc[:, 1:].values

    # Encode disease labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(diseases)

    return X, y, feature_names, label_encoder


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    n_classes: int = None,
) -> object:
    """
    Train a classification model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Either "random_forest" or "xgboost"
        n_classes: Number of classes (for XGBoost)

    Returns:
        Trained model instance
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=15,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective="multi:softprob",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict:
    """
    Evaluate model performance.

    Returns:
        Dictionary with accuracy, f1_score, and classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
    }


def save_disease_info(label_encoder: LabelEncoder) -> None:
    """Save disease metadata for API responses."""
    disease_info = {}
    for disease in label_encoder.classes_:
        # Generate placeholder ICD-10 and severity
        # In production, this would come from a medical database
        disease_info[disease] = {
            "icd10": "R69",  # Placeholder - unknown diagnosis
            "severity": "moderate",  # Default severity
        }

    with open(DISEASE_INFO_FILE, "w") as f:
        json.dump(disease_info, f, indent=2)


def main(sample_size: int = None):
    """
    Run the complete training pipeline.

    Args:
        sample_size: Optional limit on training samples (for faster iteration)
    """
    print("=" * 60)
    print("MedCareAI ML Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"    Total rows: {len(df):,}")
    print(f"    Total columns: {len(df.columns)}")

    # Prepare features and labels
    print("\n[2/6] Preparing features and labels...")
    X, y, feature_names, label_encoder = prepare_features_and_labels(df, sample_size)
    n_classes = len(label_encoder.classes_)
    print(f"    Features: {len(feature_names)}")
    print(f"    Classes: {n_classes}")
    print(f"    Samples: {len(X):,}")

    # Split data
    print("\n[3/6] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training set: {len(X_train):,} samples")
    print(f"    Test set: {len(X_test):,} samples")

    # Setup MLflow
    print("\n[4/6] Setting up MLflow tracking...")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("medcareai-disease-prediction")

    # Train and evaluate models
    best_model = None
    best_score = 0
    best_model_type = ""

    # Train only RandomForest (stable and reliable for multi-class)
    for model_type in ["random_forest"]:
        print(f"\n[5/6] Training {model_type}...")

        with mlflow.start_run(run_name=f"{model_type}_training"):
            # Train
            model = train_model(X_train, y_train, model_type, n_classes)

            # Evaluate on test set (skip CV for speed with large dataset)
            metrics = evaluate_model(model, X_test, y_test, label_encoder)
            print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Test F1 Score: {metrics['f1_score']:.4f}")

            # Log to MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_classes", n_classes)
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            mlflow.log_metric("test_f1_score", metrics["f1_score"])

            # Track best model
            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_model = model
                best_model_type = model_type

    # Save best model
    print(f"\n[6/6] Saving best model ({best_model_type})...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

    # Save feature names
    with open(FEATURE_NAMES_FILE, "w") as f:
        json.dump(feature_names, f)

    # Save disease info
    save_disease_info(label_encoder)

    print(f"    Model saved to: {MODEL_FILE}")
    print(f"    Label encoder saved to: {LABEL_ENCODER_FILE}")
    print(f"    Feature names saved to: {FEATURE_NAMES_FILE}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best model: {best_model_type}")
    print(f"Test F1 Score: {best_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MedCareAI disease prediction model")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit training samples (for faster iteration)",
    )
    args = parser.parse_args()

    main(sample_size=args.sample_size)
