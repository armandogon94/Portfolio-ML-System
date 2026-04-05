"""Shared preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 10 else None)


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns encoded df and encoder mapping."""
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train, transform both train and test."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test, scaler
