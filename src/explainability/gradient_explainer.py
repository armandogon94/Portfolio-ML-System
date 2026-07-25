"""Gradient-based explainability for PyTorch autoencoder (fraud detection)."""

import torch


class GradientExplainer:
    """Computes feature importance for a fraud autoencoder via input gradients.

    Importance = |d(reconstruction_error) / d(input_feature)|

    Higher gradient magnitude means the model is more sensitive to that feature,
    indicating it contributes more to the anomaly detection decision.
    """

    def explain(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        feature_names: list[str],
        top_n: int = 10,
    ) -> dict:
        """Compute gradient-based feature importance for a single sample.

        Args:
            model: Trained FraudAutoencoder (or any autoencoder with reconstruction_error).
            X: Float tensor of shape (1, n_features).
            feature_names: Ordered list of feature column names.
            top_n: Number of top features to include in top_features list.

        Returns:
            dict with keys:
                - feature_importances: {feature_name: importance} (non-negative floats)
                - top_features: list of {feature, importance} sorted descending
                - explanation_type: "gradient"
        """
        model.eval()

        x = X.clone().detach().requires_grad_(True)
        recon_error = model.reconstruction_error(x)  # type: ignore[operator]
        recon_error.sum().backward()

        gradient = x.grad
        if gradient is None:  # pragma: no cover - autograd always populates .grad here
            raise RuntimeError(
                "No gradient reached the input. The model must be in a graph-building "
                "state and reconstruction_error must be differentiable w.r.t. its input."
            )
        # Absolute gradient as importance
        importances = gradient.abs().squeeze(0).detach().numpy()

        feature_importances = {name: float(val) for name, val in zip(feature_names, importances)}

        sorted_features = sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)
        top_features = [
            {"feature": name, "importance": float(val)} for name, val in sorted_features[:top_n]
        ]

        return {
            "feature_importances": feature_importances,
            "top_features": top_features,
            "explanation_type": "gradient",
        }
