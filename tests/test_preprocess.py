"""Tests for shared preprocessing utilities."""

import numpy as np
import pandas as pd

from src.data.preprocess import encode_categoricals, scale_features, split_data


class TestSplitData:

    def test_split_proportions(self):
        df = pd.DataFrame({
            "a": range(100), "b": range(100),
            "target": [0] * 80 + [1] * 20,
        })
        X_train, X_test, y_train, y_test = split_data(df, "target", test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_target_not_in_features(self):
        df = pd.DataFrame({
            "a": range(20),
            "target": [0] * 14 + [1] * 6,
        })
        X_train, X_test, _, _ = split_data(df, "target", test_size=0.3)
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_reproducibility(self):
        df = pd.DataFrame({"a": range(50), "target": [0] * 40 + [1] * 10})
        X1, _, _, _ = split_data(df, "target", random_state=42)
        X2, _, _, _ = split_data(df, "target", random_state=42)
        pd.testing.assert_frame_equal(X1, X2)


class TestEncodeCategoricals:

    def test_encodes_columns(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]})
        encoded, encoders = encode_categoricals(df, ["cat"])
        assert np.issubdtype(encoded["cat"].dtype, np.integer)
        assert "cat" in encoders

    def test_preserves_numerical(self):
        df = pd.DataFrame({"cat": ["x", "y"], "num": [10, 20]})
        encoded, _ = encode_categoricals(df, ["cat"])
        assert encoded["num"].tolist() == [10, 20]

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2]})
        encoded, encoders = encode_categoricals(df, ["nonexistent"])
        assert "nonexistent" not in encoders

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"cat": ["a", "b"]})
        original = df.copy()
        encode_categoricals(df, ["cat"])
        pd.testing.assert_frame_equal(df, original)


class TestScaleFeatures:

    def test_scaled_output_shape(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"a": [1.5], "b": [4.5]})
        X_tr, X_te, scaler = scale_features(X_train, X_test, ["a", "b"])
        assert X_tr.shape == (3, 2)
        assert X_te.shape == (1, 2)

    def test_train_mean_near_zero(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_test = pd.DataFrame({"a": [3.0]})
        X_tr, _, _ = scale_features(X_train, X_test, ["a"])
        assert abs(X_tr["a"].mean()) < 1e-10

    def test_returns_scaler(self):
        from sklearn.preprocessing import StandardScaler
        X_train = pd.DataFrame({"a": [1.0, 2.0]})
        X_test = pd.DataFrame({"a": [1.5]})
        _, _, scaler = scale_features(X_train, X_test, ["a"])
        assert isinstance(scaler, StandardScaler)
