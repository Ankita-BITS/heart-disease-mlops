from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def build_preprocessor(numeric_features=None, categorical_features=None):
    """
    Builds the preprocessing pipeline for the Heart Disease dataset.

    Numerical features:
    - Missing values are imputed using the median.
    - Features are scaled using StandardScaler.

    Categorical features:
    - Missing values are imputed using the most frequent category.
    - Features are encoded using OneHotEncoder.

    The output is a ColumnTransformer that can be reused during both
    model training and inference.
    """

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
