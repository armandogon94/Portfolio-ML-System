"""Map a Kaggle Zillow-style housing DataFrame onto the canonical schema.

The canonical schema is defined by ``src/data/generate_housing.py`` and
consumed by ``src/features/housing_features.py``. Real-world vendor datasets
use a variety of column names (``SquareFootage`` vs ``sqft_living`` vs
``sqft``); this adapter collapses those variants onto a single schema so the
rest of the pipeline can stay vendor-agnostic.

Design notes (resolves SPEC.md Q1 — "how do we onboard real housing data?"):

* The adapter is best-effort: missing optional columns are imputed with sane
  defaults rather than raising, so the pipeline keeps working when a vendor
  drops a field. Numeric columns missing entirely fall back to ``0`` (with a
  warning). Numeric columns with partial NaNs fall back to the column median.
* Categorical defaults: ``has_pool`` -> 0, ``neighborhood_tier`` -> 3 (middle
  tier) per plan.md §A.1.5.
* Extra vendor columns (``Zipcode``, ``FloodZone``, ``ListingAgent``, ...)
  are dropped; only the 10 canonical columns are returned, in canonical order.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical schema — MUST match src/data/generate_housing.py output columns
# and the order expected downstream.
CANONICAL_COLUMNS: list[str] = [
    "square_feet",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size_sqft",
    "garage_spaces",
    "has_pool",
    "neighborhood_tier",
    "proximity_to_city_center",
    "price",
]

# For each canonical column, a prioritized list of source column names the
# adapter will try (first match wins). Covers Kaggle Zillow + common variants.
COLUMN_MAP: dict[str, list[str]] = {
    "square_feet":              ["SquareFootage", "living_area_sqft", "sqft_living", "sqft"],
    "bedrooms":                 ["Bedrooms", "beds", "bed"],
    "bathrooms":                ["Bathrooms", "baths", "bath"],
    "year_built":               ["YearBuilt", "year_built", "yr_built"],
    "lot_size_sqft":            ["LotSize", "lot_size_sqft", "sqft_lot"],
    "garage_spaces":            ["GarageSpaces", "garage_cars", "garage"],
    "has_pool":                 ["HasPool", "pool", "has_pool"],
    "neighborhood_tier":        ["NeighborhoodTier", "neighborhood_tier"],
    "proximity_to_city_center": ["DistanceToCBD", "proximity_to_city_center", "distance_to_center"],
    "price":                    ["SalePrice", "Price", "price"],
}

# Categorical defaults when no source column is present at all.
_CATEGORICAL_DEFAULTS: dict[str, int] = {
    "has_pool": 0,
    "neighborhood_tier": 3,  # middle tier
}


def _resolve_source_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in ``df``, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def housing_adapter(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a Zillow-style DataFrame onto the canonical housing schema.

    Parameters
    ----------
    df
        Raw vendor DataFrame. Column names may use any of the variants listed
        in ``COLUMN_MAP``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with exactly ``CANONICAL_COLUMNS`` in canonical order.
        Missing numeric columns are imputed (median or 0); missing categorical
        columns fall back to their documented defaults.
    """
    # Never mutate the caller's DataFrame.
    out = pd.DataFrame(index=df.index)
    n_rows = len(df)

    mapped: list[str] = []
    imputed: list[str] = []

    for canonical, candidates in COLUMN_MAP.items():
        source = _resolve_source_column(df, candidates)

        if source is not None:
            column = df[source]
            if column.isna().any():
                # Fill partial NaNs with the column median for numerics, or
                # the documented default for categoricals / empty columns.
                non_null = column.dropna()
                if len(non_null) > 0 and canonical not in _CATEGORICAL_DEFAULTS:
                    fill_value = non_null.median()
                elif canonical in _CATEGORICAL_DEFAULTS:
                    fill_value = _CATEGORICAL_DEFAULTS[canonical]
                else:
                    fill_value = 0
                # Coerce to numeric first to avoid pandas' object-downcast
                # FutureWarning when the source column is all-NaN and
                # therefore inferred as object dtype.
                column = pd.to_numeric(column, errors="coerce").fillna(fill_value)
                imputed.append(canonical)
            out[canonical] = column.values
            mapped.append(f"{canonical}<-{source}")
            continue

        # No source column matched — impute.
        if canonical in _CATEGORICAL_DEFAULTS:
            out[canonical] = _CATEGORICAL_DEFAULTS[canonical]
        else:
            # Numeric column fully absent — best we can do is 0.
            logger.warning(
                "housing_adapter: no source column for %r; filling with 0. "
                "Candidates tried: %s",
                canonical,
                candidates,
            )
            out[canonical] = 0
        imputed.append(canonical)

        # Keep row count consistent with the input even when we fabricate.
        if n_rows == 0:
            out[canonical] = pd.Series(dtype="float64")

    logger.info(
        "housing_adapter: mapped=%s imputed=%s rows=%d",
        mapped,
        imputed,
        n_rows,
    )

    # Enforce canonical column order.
    return out[CANONICAL_COLUMNS]
