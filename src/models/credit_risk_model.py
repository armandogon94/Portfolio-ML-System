"""XGBoost credit risk scoring model."""

import xgboost as xgb


def create_model(params: dict) -> xgb.XGBClassifier:
    """Create XGBoost classifier with given parameters."""
    return xgb.XGBClassifier(
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 200),
        objective=params.get("objective", "binary:logistic"),
        eval_metric=params.get("eval_metric", "auc"),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        random_state=42,
        verbosity=0,
    )
