import pandas as pd
from sklearn.compose import ColumnTransformer

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_can_fit_transform_sample_data():
    sample_data = pd.DataFrame(
        {
            "age": [63, 45, None],
            "trestbps": [145, 130, 120],
            "chol": [233, 250, None],
            "thalach": [150, 187, 172],
            "oldpeak": [2.3, 3.5, 1.4],
            "sex": [1, 0, 1],
            "cp": [3, 2, 1],
            "fbs": [1, 0, 0],
            "restecg": [0, 1, 0],
            "exang": [0, 0, 1],
            "slope": [0, 2, 1],
            "ca": [0, 1, None],
            "thal": [1, 2, 3],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    transformed_data = preprocessor.fit_transform(sample_data)

    assert transformed_data.shape[0] == sample_data.shape[0]
    assert transformed_data.shape[1] > 0
