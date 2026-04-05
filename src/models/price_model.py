"""LightGBM real estate price prediction model."""

import lightgbm as lgb


def create_model(params: dict) -> lgb.LGBMRegressor:
    """Create LightGBM regressor with given parameters."""
    return lgb.LGBMRegressor(
        objective=params.get("objective", "regression"),
        metric=params.get("metric", "rmse"),
        num_leaves=params.get("num_leaves", 63),
        learning_rate=params.get("learning_rate", 0.05),
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", -1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        verbose=-1,
        random_state=42,
    )
