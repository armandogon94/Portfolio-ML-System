"""Root conftest - import xgboost before torch to avoid libomp conflict on macOS."""

import xgboost  # noqa: F401
import lightgbm  # noqa: F401
