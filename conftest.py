"""Root conftest - import xgboost before torch to avoid libomp conflict on macOS."""

import lightgbm  # noqa: F401
import xgboost  # noqa: F401
