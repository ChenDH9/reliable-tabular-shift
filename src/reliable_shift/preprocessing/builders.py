from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def prepare_features(
    frame: pd.DataFrame, feature_columns: list[str], numeric_columns: list[str]
) -> pd.DataFrame:
    result = frame.loc[:, feature_columns].copy()
    numeric = set(numeric_columns)
    for column in feature_columns:
        if column in numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        else:
            result[column] = result[column].astype("string").fillna("__MISSING__")
    return result


def build_preprocessor(
    *,
    feature_columns: list[str],
    numeric_columns: list[str],
    sparse_categorical: bool,
) -> ColumnTransformer:
    numeric = [column for column in feature_columns if column in set(numeric_columns)]
    categorical = [column for column in feature_columns if column not in set(numeric_columns)]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    if sparse_categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", encoder, categorical)],
        remainder="drop",
        sparse_threshold=0.3 if sparse_categorical else 0.0,
    )
