"""Adapters that map third-party datasets to the project's canonical schemas.

Each adapter takes a raw vendor DataFrame (e.g., Kaggle Zillow listings) and
returns a DataFrame with the project's canonical column names, dtypes, and
order — ready for the corresponding feature engineering pipeline.
"""

from src.data.adapters.housing_adapter import housing_adapter

__all__ = ["housing_adapter"]
