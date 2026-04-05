"""Shared evaluation utilities."""

import pandas as pd
from pathlib import Path
from src.config import get_project_root


def save_comparison_summary() -> Path:
    """Combine all per-model CSV results into a single summary."""
    results_dir = get_project_root() / "results"
    csv_files = list(results_dir.glob("*_metrics.csv"))

    if not csv_files:
        return results_dir / "comparison_summary.csv"

    rows = []
    for f in csv_files:
        problem = f.stem.replace("_metrics", "")
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            rows.append({"problem": problem, "metric": row["metric"], "value": row["value"]})

    summary = pd.DataFrame(rows)
    summary_path = results_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path
