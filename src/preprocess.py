"""
preprocess.py — Load creditcard.csv, scale Amount/Time, apply SMOTE, split into train/test.

Returns X_train, X_test, y_train, y_test ready for model training.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows | Fraud rate: {df['Class'].mean():.4%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Hour from Time and drop the raw Time column."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df.drop(columns=["Time"], inplace=True)
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScaler applied only to Amount and Hour.
    PCA features V1–V28 are already scaled by the dataset provider.
    """
    scaler = StandardScaler()
    df = df.copy()
    df[["Amount", "Hour"]] = scaler.fit_transform(df[["Amount", "Hour"]])
    return df, scaler


def split_and_resample(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    smote_random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Stratified train/test split (keeps class ratio in test set).
    2. SMOTE applied only to the training set to avoid data leakage.
    """
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    feature_names = df.drop(columns=["Class"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Pre-SMOTE  | Train fraud: {y_train.sum():,} / {len(y_train):,}")
    smote = SMOTE(random_state=smote_random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE | Train fraud: {y_train_res.sum():,} / {len(y_train_res):,}")

    return X_train_res, X_test, y_train_res, y_test, feature_names


def save_scaler(scaler: StandardScaler, path: str | None = None) -> None:
    path = path or os.path.join(MODELS_DIR, "scaler.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {path}")


def run(data_path: str = DATA_PATH) -> tuple:
    df = load_data(data_path)
    df = engineer_features(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_resample(df)
    save_scaler(scaler)
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, scaler = run()
    print(f"\nFinal shapes → X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Features: {feature_names}")
