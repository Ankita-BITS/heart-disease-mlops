from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/heart_disease_raw.csv")
PROCESSED_DATA_PATH = Path("data/processed/heart_disease_clean.csv")


def download_heart_disease_data():
    """
    Downloads the Heart Disease dataset from the UCI Machine Learning Repository.
    The target is converted into a binary classification label:
    0 = no heart disease
    1 = presence of heart disease
    """

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)

    # Standardize target column name
    target_col = y.columns[0]
    df = df.rename(columns={target_col: "target"})

    # Convert target to binary:
    # UCI target values: 0 = no disease, 1-4 = disease presence
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_DATA_PATH, index=False)

    # Basic cleaning: remove duplicate rows
    df = df.drop_duplicates()

    # Save cleaned version for EDA/modeling
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Dataset downloaded successfully.")
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")
    print(f"Clean dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    download_heart_disease_data()
