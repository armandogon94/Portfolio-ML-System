"""Tests for model explainability modules.

Tests SHAP (tree models) and gradient-based (autoencoder) explainers,
plus API /explain/* endpoints.
"""

import numpy as np

from src.features.credit_risk_features import (
    engineer_features as credit_features,
)
from src.features.credit_risk_features import (
    get_feature_columns as credit_feature_cols,
)
from src.features.fraud_features import (
    engineer_features as fraud_engineer,
)
from src.features.fraud_features import (
    get_feature_columns as fraud_feature_cols,
)
from src.features.housing_features import (
    engineer_features as housing_features,
)
from src.features.housing_features import (
    get_feature_columns as housing_feature_cols,
)

# ── Task 4.1: SHAP Explainer ─────────────────────────────────────────────────


class TestSHAPExplainer:
    """Tests for SHAPExplainer (XGBoost + LightGBM)."""

    def test_shap_explainer_imports(self):
        """SHAPExplainer can be imported from explainability module."""
        from src.explainability.shap_explainer import SHAPExplainer  # noqa: F401

    def test_explain_credit_risk_returns_dict(self, tiny_credit_risk_model, credit_risk_df):
        """explain() returns a dict with feature_importances and top_features."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        assert isinstance(result, dict)
        assert "feature_importances" in result
        assert "top_features" in result
        assert "explanation_type" in result

    def test_explain_credit_risk_feature_names_match(self, tiny_credit_risk_model, credit_risk_df):
        """feature_importances keys match the provided feature names."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        assert set(result["feature_importances"].keys()) == set(credit_feature_cols())

    def test_explain_credit_risk_values_are_floats(self, tiny_credit_risk_model, credit_risk_df):
        """feature_importances values are all floats."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        for v in result["feature_importances"].values():
            assert isinstance(v, float)

    def test_explain_top_features_sorted_by_abs_importance(
        self, tiny_credit_risk_model, credit_risk_df
    ):
        """top_features list is sorted descending by |shap_value|."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        top = result["top_features"]
        assert len(top) > 0
        # Each entry is {"feature": ..., "importance": ...}
        importances = [abs(entry["importance"]) for entry in top]
        assert importances == sorted(importances, reverse=True)

    def test_explain_top_features_default_top_n(self, tiny_credit_risk_model, credit_risk_df):
        """top_features returns at most top_n=10 features by default."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        assert len(result["top_features"]) <= 10

    def test_explain_type_is_shap(self, tiny_credit_risk_model, credit_risk_df):
        """explanation_type field is 'shap'."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols())

        assert result["explanation_type"] == "shap"

    def test_explain_price_lightgbm(self, tiny_price_model, housing_df):
        """explain() works for LightGBM price model."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = housing_features(housing_df.copy())
        X = df[housing_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_price_model, sample, housing_feature_cols())

        assert set(result["feature_importances"].keys()) == set(housing_feature_cols())
        assert result["explanation_type"] == "shap"
        assert len(result["top_features"]) > 0

    def test_explain_custom_top_n(self, tiny_credit_risk_model, credit_risk_df):
        """top_n parameter limits number of top_features returned."""
        from src.explainability.shap_explainer import SHAPExplainer

        df = credit_features(credit_risk_df.copy())
        X = df[credit_feature_cols()]
        sample = X.iloc[:1]

        explainer = SHAPExplainer()
        result = explainer.explain(tiny_credit_risk_model, sample, credit_feature_cols(), top_n=3)

        assert len(result["top_features"]) == 3


# ── Task 4.2: Gradient Explainer ─────────────────────────────────────────────


class TestGradientExplainer:
    """Tests for GradientExplainer (fraud autoencoder)."""

    def test_gradient_explainer_imports(self):
        """GradientExplainer can be imported."""
        from src.explainability.gradient_explainer import GradientExplainer  # noqa: F401

    def test_explain_fraud_returns_dict(self, tiny_fraud_artifacts, fraud_df):
        """explain() returns dict with feature_importances and top_features."""
        import torch

        from src.explainability.gradient_explainer import GradientExplainer

        df, _ = fraud_engineer(fraud_df.copy())
        X = df[fraud_feature_cols()].values[:1].astype(np.float32)
        scaler = tiny_fraud_artifacts["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        model = tiny_fraud_artifacts["autoencoder"]
        explainer = GradientExplainer()
        result = explainer.explain(model, X_tensor, fraud_feature_cols())

        assert isinstance(result, dict)
        assert "feature_importances" in result
        assert "top_features" in result
        assert "explanation_type" in result

    def test_explain_fraud_feature_names_match(self, tiny_fraud_artifacts, fraud_df):
        """feature_importances keys match fraud feature columns."""
        import torch

        from src.explainability.gradient_explainer import GradientExplainer

        df, _ = fraud_engineer(fraud_df.copy())
        X = df[fraud_feature_cols()].values[:1].astype(np.float32)
        scaler = tiny_fraud_artifacts["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        model = tiny_fraud_artifacts["autoencoder"]
        explainer = GradientExplainer()
        result = explainer.explain(model, X_tensor, fraud_feature_cols())

        assert set(result["feature_importances"].keys()) == set(fraud_feature_cols())

    def test_explain_fraud_importances_are_non_negative(self, tiny_fraud_artifacts, fraud_df):
        """Gradient importances are absolute values (non-negative)."""
        import torch

        from src.explainability.gradient_explainer import GradientExplainer

        df, _ = fraud_engineer(fraud_df.copy())
        X = df[fraud_feature_cols()].values[:1].astype(np.float32)
        scaler = tiny_fraud_artifacts["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        model = tiny_fraud_artifacts["autoencoder"]
        explainer = GradientExplainer()
        result = explainer.explain(model, X_tensor, fraud_feature_cols())

        for v in result["feature_importances"].values():
            assert v >= 0.0

    def test_explain_fraud_top_features_sorted(self, tiny_fraud_artifacts, fraud_df):
        """top_features sorted descending by importance."""
        import torch

        from src.explainability.gradient_explainer import GradientExplainer

        df, _ = fraud_engineer(fraud_df.copy())
        X = df[fraud_feature_cols()].values[:1].astype(np.float32)
        scaler = tiny_fraud_artifacts["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        model = tiny_fraud_artifacts["autoencoder"]
        explainer = GradientExplainer()
        result = explainer.explain(model, X_tensor, fraud_feature_cols())

        importances = [e["importance"] for e in result["top_features"]]
        assert importances == sorted(importances, reverse=True)

    def test_explain_fraud_type_is_gradient(self, tiny_fraud_artifacts, fraud_df):
        """explanation_type is 'gradient'."""
        import torch

        from src.explainability.gradient_explainer import GradientExplainer

        df, _ = fraud_engineer(fraud_df.copy())
        X = df[fraud_feature_cols()].values[:1].astype(np.float32)
        scaler = tiny_fraud_artifacts["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        model = tiny_fraud_artifacts["autoencoder"]
        explainer = GradientExplainer()
        result = explainer.explain(model, X_tensor, fraud_feature_cols())

        assert result["explanation_type"] == "gradient"


# ── Task 4.3: API /explain/* endpoints ───────────────────────────────────────


class TestExplainAPIEndpoints:
    """Tests for /explain/credit-risk, /explain/price, /explain/fraud."""

    def test_explain_credit_risk_returns_200(self, api_client):
        """POST /explain/credit-risk returns 200."""
        resp = api_client.post("/explain/credit-risk", json={})
        assert resp.status_code == 200

    def test_explain_credit_risk_response_structure(self, api_client):
        """POST /explain/credit-risk response has correct fields."""
        resp = api_client.post("/explain/credit-risk", json={})
        data = resp.json()

        assert "feature_importances" in data
        assert "top_features" in data
        assert "explanation_type" in data
        assert data["explanation_type"] == "shap"

    def test_explain_credit_risk_feature_names_present(self, api_client):
        """feature_importances keys are credit risk feature names."""
        resp = api_client.post("/explain/credit-risk", json={})
        data = resp.json()

        assert set(data["feature_importances"].keys()) == set(credit_feature_cols())

    def test_explain_price_returns_200(self, api_client):
        """POST /explain/price returns 200."""
        resp = api_client.post("/explain/price", json={})
        assert resp.status_code == 200

    def test_explain_price_response_structure(self, api_client):
        """POST /explain/price response has correct fields."""
        resp = api_client.post("/explain/price", json={})
        data = resp.json()

        assert "feature_importances" in data
        assert "top_features" in data
        assert "explanation_type" in data
        assert data["explanation_type"] == "shap"

    def test_explain_price_feature_names_present(self, api_client):
        """feature_importances keys are housing feature names."""
        resp = api_client.post("/explain/price", json={})
        data = resp.json()

        assert set(data["feature_importances"].keys()) == set(housing_feature_cols())

    def test_explain_fraud_returns_200(self, api_client):
        """POST /explain/fraud returns 200."""
        resp = api_client.post("/explain/fraud", json={})
        assert resp.status_code == 200

    def test_explain_fraud_response_structure(self, api_client):
        """POST /explain/fraud response has correct fields."""
        resp = api_client.post("/explain/fraud", json={})
        data = resp.json()

        assert "feature_importances" in data
        assert "top_features" in data
        assert "explanation_type" in data
        assert data["explanation_type"] == "gradient"

    def test_explain_fraud_feature_names_present(self, api_client):
        """feature_importances keys are fraud feature names."""
        resp = api_client.post("/explain/fraud", json={})
        data = resp.json()

        assert set(data["feature_importances"].keys()) == set(fraud_feature_cols())

    def test_explain_top_features_is_list(self, api_client):
        """top_features is a list of {feature, importance} dicts."""
        resp = api_client.post("/explain/credit-risk", json={})
        data = resp.json()

        top = data["top_features"]
        assert isinstance(top, list)
        assert len(top) > 0
        assert "feature" in top[0]
        assert "importance" in top[0]
