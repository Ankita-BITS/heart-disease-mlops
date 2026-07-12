from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/heart_disease_clean.csv")


def test_clean_dataset_exists():
    assert DATA_PATH.exists(), "Cleaned dataset file does not exist."


def test_clean_dataset_has_target_column():
    df = pd.read_csv(DATA_PATH)

    assert "target" in df.columns, "Target column is missing from dataset."


def test_target_is_binary():
    df = pd.read_csv(DATA_PATH)

    unique_targets = set(df["target"].unique())

    assert unique_targets.issubset({0, 1}), (
        f"Target column should only contain 0 and 1, found: {unique_targets}"
    )


def test_dataset_has_records():
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0, "Dataset should contain at least one record."
