import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "model" / "heart_disease_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "model" / "feature_metadata.json"


def load_model():
    """
    Load the saved sklearn pipeline.

    The saved pipeline includes both preprocessing and the trained classifier.
    This ensures that the same transformation logic is used during inference
    as was used during training.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run src/train.py before running prediction."
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    """
    Load feature metadata used to validate and order prediction inputs.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata file not found at {METADATA_PATH}. "
            "Run src/train.py before running prediction."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_input(input_path):
    """
    Load prediction input from a JSON file.

    The input JSON may be either:
    - a single JSON object
    - a list of JSON objects
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a dictionary or a list of dictionaries.")

    return pd.DataFrame(data)


def validate_and_order_features(input_df, metadata):
    """
    Validate that required model features are present and order columns correctly.
    """

    required_features = metadata["numeric_features"] + metadata["categorical_features"]

    missing_features = [
        feature for feature in required_features if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required input features: {missing_features}")

    return input_df[required_features]


def make_predictions(input_path):
    """
    Generate predictions and confidence scores from a JSON input file.
    """

    model = load_model()
    metadata = load_metadata()

    input_df = load_input(input_path)
    input_df = validate_and_order_features(input_df, metadata)

    predictions = model.predict(input_df)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[:, 1]
    else:
        probabilities = [None] * len(predictions)

    results = []

    for prediction, probability in zip(predictions, probabilities):
        results.append(
            {
                "prediction": int(prediction),
                "prediction_label": (
                    "heart_disease_present"
                    if int(prediction) == 1
                    else "heart_disease_absent"
                ),
                "confidence_probability": (
                    round(float(probability), 4)
                    if probability is not None
                    else None
                ),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run heart disease prediction using the saved model pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file.",
    )

    args = parser.parse_args()

    results = make_predictions(args.input)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
