"""Tests for src/data/modality.py — modality dispatcher.

The dispatcher routes data loading across three modalities:
``synthetic``, ``stream``, and ``mixed``. External dependencies
(``kaggle_cached``, ``pd.read_csv``) are mocked so these tests run
offline and without real Kaggle credentials.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


class TestLoadForModalitySynthetic:

    def test_load_for_modality_synthetic(self, monkeypatch):
        """With modality='synthetic', the synthetic_loader is invoked and its
        return value is passed through unchanged. Nothing else is touched."""
        synthetic_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        calls = []

        def fake_loader():
            calls.append("called")
            return synthetic_df

        # Sentinel: if the stream path is accidentally taken, blow up loudly.
        def boom(*_args, **_kwargs):
            raise AssertionError("kaggle_cached must not be called for synthetic modality")

        monkeypatch.setattr("src.data.modality.kaggle_cached", boom)

        from src.data.modality import load_for_modality

        result = load_for_modality("synthetic", synthetic_loader=fake_loader)

        assert calls == ["called"]
        pd.testing.assert_frame_equal(result, synthetic_df)


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


class TestLoadForModalityStream:

    def test_load_for_modality_stream(self, monkeypatch, tmp_path):
        """With modality='stream', kaggle_cached is called with slug+filename,
        pd.read_csv reads the returned path, and stream_adapter transforms it."""
        fake_csv = tmp_path / "raw.csv"
        fake_csv.write_text("not,real,csv\n")

        captured = {}

        def fake_kaggle_cached(slug, *, filename=None):
            captured["slug"] = slug
            captured["filename"] = filename
            return fake_csv

        raw_df = pd.DataFrame({"x": [10, 20, 30], "junk_col": ["a", "b", "c"]})

        def fake_read_csv(path):
            captured["read_path"] = Path(path)
            return raw_df

        def adapter(df: pd.DataFrame) -> pd.DataFrame:
            # Adapter drops a column and renames one — we assert the effect.
            out = df.drop(columns=["junk_col"]).rename(columns={"x": "amount"})
            captured["adapter_called"] = True
            return out

        monkeypatch.setattr("src.data.modality.kaggle_cached", fake_kaggle_cached)
        monkeypatch.setattr("src.data.modality.pd.read_csv", fake_read_csv)

        from src.data.modality import load_for_modality

        def unused_loader():
            raise AssertionError("synthetic_loader must not be called for stream modality")

        result = load_for_modality(
            "stream",
            synthetic_loader=unused_loader,
            stream_slug="owner/some-dataset",
            stream_file="raw.csv",
            stream_adapter=adapter,
        )

        assert captured["slug"] == "owner/some-dataset"
        assert captured["filename"] == "raw.csv"
        assert captured["read_path"] == fake_csv
        assert captured["adapter_called"] is True
        assert list(result.columns) == ["amount"]
        assert result["amount"].tolist() == [10, 20, 30]

    def test_load_for_modality_stream_without_adapter(self, monkeypatch, tmp_path):
        """When stream_adapter is None, the raw CSV DataFrame is returned unchanged."""
        fake_csv = tmp_path / "raw.csv"
        fake_csv.write_text("col\n1\n")

        raw_df = pd.DataFrame({"col": [1, 2, 3]})

        monkeypatch.setattr(
            "src.data.modality.kaggle_cached",
            lambda slug, *, filename=None: fake_csv,
        )
        monkeypatch.setattr("src.data.modality.pd.read_csv", lambda _path: raw_df)

        from src.data.modality import load_for_modality

        result = load_for_modality(
            "stream",
            synthetic_loader=lambda: pd.DataFrame(),
            stream_slug="owner/dataset",
            stream_file="raw.csv",
            # stream_adapter omitted — optional
        )

        pd.testing.assert_frame_equal(result, raw_df)


# ---------------------------------------------------------------------------
# mixed
# ---------------------------------------------------------------------------


class TestLoadForModalityMixed:

    def test_load_for_modality_mixed(self, monkeypatch, tmp_path):
        """With modality='mixed', synthetic + stream rows are concatenated
        and tagged with a 'modality' column labelled per-row."""
        synthetic_df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
        stream_raw = pd.DataFrame({"amount": [10.0, 20.0]})

        fake_csv = tmp_path / "raw.csv"
        fake_csv.write_text("amount\n")

        monkeypatch.setattr(
            "src.data.modality.kaggle_cached",
            lambda slug, *, filename=None: fake_csv,
        )
        monkeypatch.setattr("src.data.modality.pd.read_csv", lambda _p: stream_raw)

        from src.data.modality import load_for_modality

        result = load_for_modality(
            "mixed",
            synthetic_loader=lambda: synthetic_df,
            stream_slug="owner/dataset",
            stream_file="raw.csv",
        )

        # Total row count = synthetic + stream
        assert len(result) == len(synthetic_df) + len(stream_raw)
        # modality column exists with both values
        assert "modality" in result.columns
        assert set(result["modality"].unique()) == {"synthetic", "stream"}
        # Per-row labels: first N synthetic, next M stream
        assert result["modality"].tolist() == (
            ["synthetic"] * len(synthetic_df) + ["stream"] * len(stream_raw)
        )
        # Payload column preserved
        assert result["amount"].tolist() == [1.0, 2.0, 3.0, 10.0, 20.0]

    def test_load_for_modality_mixed_applies_stream_adapter(self, monkeypatch, tmp_path):
        """In 'mixed' mode, stream_adapter is applied to the stream rows
        before concatenation — matching 'stream' mode semantics."""
        synthetic_df = pd.DataFrame({"amount": [1.0]})
        raw_stream = pd.DataFrame({"x": [42.0, 99.0]})

        fake_csv = tmp_path / "raw.csv"
        fake_csv.write_text("x\n")

        monkeypatch.setattr(
            "src.data.modality.kaggle_cached",
            lambda slug, *, filename=None: fake_csv,
        )
        monkeypatch.setattr("src.data.modality.pd.read_csv", lambda _p: raw_stream)

        def adapter(df):
            return df.rename(columns={"x": "amount"})

        from src.data.modality import load_for_modality

        result = load_for_modality(
            "mixed",
            synthetic_loader=lambda: synthetic_df,
            stream_slug="owner/dataset",
            stream_file="raw.csv",
            stream_adapter=adapter,
        )

        assert result["amount"].tolist() == [1.0, 42.0, 99.0]
        assert result["modality"].tolist() == ["synthetic", "stream", "stream"]


# ---------------------------------------------------------------------------
# invalid modality
# ---------------------------------------------------------------------------


class TestLoadForModalityInvalid:

    def test_load_for_modality_invalid_raises(self):
        """Unknown modality raises ValueError whose message lists the valid
        modalities so callers can discover them from the error."""
        from src.data.modality import load_for_modality

        with pytest.raises(ValueError) as excinfo:
            load_for_modality(
                "invalid",
                synthetic_loader=lambda: pd.DataFrame(),
            )

        msg = str(excinfo.value)
        # All three valid modalities must be named in the error message.
        assert "synthetic" in msg
        assert "stream" in msg
        assert "mixed" in msg

    def test_load_for_modality_stream_without_slug_raises(self):
        """'stream' and 'mixed' modalities require stream_slug — surface a
        clear error rather than a cryptic AttributeError downstream."""
        from src.data.modality import load_for_modality

        with pytest.raises(ValueError):
            load_for_modality(
                "stream",
                synthetic_loader=lambda: pd.DataFrame(),
                stream_slug=None,
            )
