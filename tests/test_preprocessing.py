from __future__ import annotations

import pandas as pd

from reliable_shift.preprocessing import build_preprocessor, prepare_features


def test_preprocessing_fit_on_training_only() -> None:
    train = pd.DataFrame({"numeric": [1.0, None, 3.0], "category": ["a", "b", "a"]})
    target = pd.DataFrame({"numeric": [1000.0], "category": ["target_only"]})
    preprocessor = build_preprocessor(
        feature_columns=["numeric", "category"],
        numeric_columns=["numeric"],
        sparse_categorical=True,
    )
    preprocessor.fit(prepare_features(train, ["numeric", "category"], ["numeric"]))
    encoder = preprocessor.named_transformers_["categorical"]
    assert "target_only" not in set(encoder.categories_[0])
    transformed = preprocessor.transform(
        prepare_features(target, ["numeric", "category"], ["numeric"])
    )
    assert transformed.shape[0] == 1


def test_missing_values_are_supported() -> None:
    frame = pd.DataFrame({"numeric": [None, 2.0], "category": [None, "x"]})
    prepared = prepare_features(frame, ["numeric", "category"], ["numeric"])
    assert prepared.category.iloc[0] == "__MISSING__"
