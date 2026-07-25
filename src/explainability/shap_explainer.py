"""SHAP-based explainability for tree models (XGBoost, LightGBM)."""

import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """Computes SHAP values for XGBoost and LightGBM tree models.

    Uses TreeExplainer for fast, exact SHAP computation.
    """

    def explain(
        self,
        model,
        X: pd.DataFrame,
        feature_names: list[str],
        top_n: int = 10,
    ) -> dict:
        """Compute SHAP values for a single sample.

        Args:
            model: Trained XGBoost or LightGBM model.
            X: DataFrame with a single row (the sample to explain).
            feature_names: Ordered list of feature column names.
            top_n: Number of top features to include in top_features list.

        Returns:
            dict with keys:
                - feature_importances: {feature_name: shap_value} for all features
                - top_features: list of {feature, importance} sorted by |shap_value| desc
                - explanation_type: "shap"
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification (XGBoost), shap_values may be 2D (n_samples, n_features)
        # For LightGBM regression, same shape. Take the first (and only) sample.
        if isinstance(shap_values, list):
            # Multi-class or binary with separate arrays per class — use class 1
            values = np.array(shap_values[1]).flatten()
        else:
            values = np.array(shap_values).flatten()

        feature_importances = {name: float(val) for name, val in zip(feature_names, values)}

        sorted_features = sorted(
            feature_importances.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top_features = [
            {"feature": name, "importance": float(val)} for name, val in sorted_features[:top_n]
        ]

        return {
            "feature_importances": feature_importances,
            "top_features": top_features,
            "explanation_type": "shap",
        }
