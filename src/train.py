import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


RANDOM_STATE = 42

DATA_PATH = Path("data/processed/heart_disease_clean.csv")
MODEL_DIR = Path("model")
REPORTS_DIR = Path("reports")
FIGURE_DIR = Path("reports/figures")

TARGET_COLUMN = "target"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run src/download_data.py before training."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    return df


def get_available_features(df):
    numeric_features = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    missing_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(
        numeric_features + categorical_features
    )

    if missing_features:
        print(f"Warning: These expected features were not found: {missing_features}")

    return numeric_features, categorical_features


def build_model_pipeline(classifier, numeric_features, categorical_features):
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def get_model_configs(numeric_features, categorical_features):
    logistic_regression_pipeline = build_model_pipeline(
        classifier=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    random_forest_pipeline = build_model_pipeline(
        classifier=RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model_configs = {
        "logistic_regression": {
            "pipeline": logistic_regression_pipeline,
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        },
        "random_forest": {
            "pipeline": random_forest_pipeline,
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
        },
    }

    return model_configs


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probability),
    }

    return metrics, y_pred, y_probability


def save_evaluation_plots(model, X_test, y_test, y_pred, model_name):
    confusion = confusion_matrix(y_test, y_pred)

    confusion_matrix_path = FIGURE_DIR / f"{model_name}_confusion_matrix.png"
    roc_curve_path = FIGURE_DIR / f"{model_name}_roc_curve.png"

    ConfusionMatrixDisplay(confusion_matrix=confusion).plot()
    plt.title(f"{model_name.replace('_', ' ').title()} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name.replace('_', ' ').title()} - ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()

    return confusion_matrix_path, roc_curve_path


def train_and_compare_models():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart_disease_classification")

    df = load_dataset()

    numeric_features, categorical_features = get_available_features(df)

    X = df[numeric_features + categorical_features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_configs = get_model_configs(numeric_features, categorical_features)

    results = []
    trained_models = {}

    for model_name, config in model_configs.items():
        print(f"\nTraining model: {model_name}")

        with mlflow.start_run(run_name=model_name):
            grid_search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                scoring="roc_auc",
                cv=cv_strategy,
                n_jobs=-1,
                refit=True,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            metrics, y_pred, _ = evaluate_model(best_model, X_test, y_test)

            confusion_matrix_path, roc_curve_path = save_evaluation_plots(
                model=best_model,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                model_name=model_name,
            )

            result = {
                "model_name": model_name,
                "best_cv_roc_auc": grid_search.best_score_,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1_score": metrics["f1_score"],
                "test_roc_auc": metrics["roc_auc"],
                "best_params": grid_search.best_params_,
            }

            results.append(result)
            trained_models[model_name] = best_model

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)

            for param_name, param_value in grid_search.best_params_.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            mlflow.log_metric("test_precision", metrics["precision"])
            mlflow.log_metric("test_recall", metrics["recall"])
            mlflow.log_metric("test_f1_score", metrics["f1_score"])
            mlflow.log_metric("test_roc_auc", metrics["roc_auc"])

            mlflow.log_artifact(str(confusion_matrix_path))
            mlflow.log_artifact(str(roc_curve_path))

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                input_example=X_test.iloc[[0]],
                serialization_format="cloudpickle",
            )

            print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1-score: {metrics['f1_score']:.4f}")
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Best Params: {grid_search.best_params_}")

    results_df = pd.DataFrame(results)
    results_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_model_row = results_df.sort_values(
        by="test_roc_auc",
        ascending=False,
    ).iloc[0]

    best_model_name = best_model_row["model_name"]
    final_model = trained_models[best_model_name]

    joblib.dump(final_model, MODEL_DIR / "heart_disease_pipeline.joblib")

    sample_input = X_test.iloc[[0]]
    sample_input.to_json(
        MODEL_DIR / "sample_input.json",
        orient="records",
        indent=4,
    )

    feature_metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "selected_model": best_model_name,
    }

    with open(MODEL_DIR / "feature_metadata.json", "w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, indent=4)

    with mlflow.start_run(run_name="final_selected_model"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_metric(
            "selected_model_test_roc_auc",
            float(best_model_row["test_roc_auc"]),
        )
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(MODEL_DIR / "feature_metadata.json"))

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="final_model",
            input_example=sample_input,
            serialization_format="cloudpickle",
        )

    print("\nModel comparison saved to reports/model_comparison.csv")
    print(f"Best model selected: {best_model_name}")
    print("Final pipeline saved to model/heart_disease_pipeline.joblib")
    print("Sample input saved to model/sample_input.json")
    print("MLflow experiment tracking completed.")


if __name__ == "__main__":
    train_and_compare_models()
