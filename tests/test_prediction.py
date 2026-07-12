from pathlib import Path

from predict import make_predictions


MODEL_PATH = Path("model/heart_disease_pipeline.joblib")
SAMPLE_INPUT_PATH = Path("model/sample_input.json")


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Trained model file does not exist."


def test_sample_input_exists():
    assert SAMPLE_INPUT_PATH.exists(), "Sample input JSON file does not exist."


def test_prediction_output_format():
    results = make_predictions(SAMPLE_INPUT_PATH)

    assert isinstance(results, list)
    assert len(results) > 0

    first_result = results[0]

    assert "prediction" in first_result
    assert "prediction_label" in first_result
    assert "confidence_probability" in first_result

    assert first_result["prediction"] in [0, 1]
    assert first_result["prediction_label"] in [
        "heart_disease_absent",
        "heart_disease_present",
    ]

    assert 0.0 <= first_result["confidence_probability"] <= 1.0
