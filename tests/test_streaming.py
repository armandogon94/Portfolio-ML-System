"""Tests for src/data/stream.py — Kaggle + HF streaming primitives.

All external libs (kagglehub, datasets) are mocked. Network tests live
under @pytest.mark.network and are skipped in the default suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# kaggle_cached
# ---------------------------------------------------------------------------


class TestKaggleCached:

    def test_kaggle_cached_returns_path(self, monkeypatch, tmp_path):
        """Returns a pathlib.Path to the cached dataset directory."""
        fake_cache = tmp_path / "cached_dataset"
        fake_cache.mkdir()

        fake_kagglehub = SimpleNamespace(
            dataset_download=lambda slug: str(fake_cache)
        )
        monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
        # Skip credential lookup during unit tests
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        from src.data.stream import kaggle_cached

        result = kaggle_cached("owner/dataset-slug")

        assert isinstance(result, Path)
        assert result == fake_cache

    def test_kaggle_cached_with_filename(self, monkeypatch, tmp_path):
        """With filename kwarg, returns path to the specific file inside the dataset."""
        fake_cache = tmp_path / "cached_dataset"
        fake_cache.mkdir()
        (fake_cache / "data.csv").write_text("col1,col2\n1,2\n")

        fake_kagglehub = SimpleNamespace(
            dataset_download=lambda slug: str(fake_cache)
        )
        monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        from src.data.stream import kaggle_cached

        result = kaggle_cached("owner/dataset-slug", filename="data.csv")

        assert result == fake_cache / "data.csv"
        assert result.is_file()

    def test_kaggle_cached_ensures_credentials_first(self, monkeypatch, tmp_path):
        """kaggle_cached calls ensure_kaggle_env() before touching kagglehub."""
        fake_cache = tmp_path / "cached_dataset"
        fake_cache.mkdir()

        calls = []
        fake_kagglehub = SimpleNamespace(
            dataset_download=lambda slug: (calls.append(("dl", slug)), str(fake_cache))[1]
        )
        monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env",
            lambda: calls.append(("ensure", None)),
        )

        from src.data.stream import kaggle_cached

        kaggle_cached("owner/dataset")

        # ensure_kaggle_env is called before kagglehub.dataset_download
        assert [c[0] for c in calls] == ["ensure", "dl"]


# ---------------------------------------------------------------------------
# hf_stream
# ---------------------------------------------------------------------------


class TestHfStream:

    def test_hf_stream_yields_rows(self, monkeypatch):
        """Wraps datasets.load_dataset(..., streaming=True) and yields each row."""
        fake_rows = [
            {"text": "row one", "label": 0},
            {"text": "row two", "label": 1},
            {"text": "row three", "label": 0},
        ]

        captured_kwargs = {}

        def fake_load_dataset(dataset_id, split, streaming):
            captured_kwargs["dataset_id"] = dataset_id
            captured_kwargs["split"] = split
            captured_kwargs["streaming"] = streaming
            return iter(fake_rows)

        fake_datasets = SimpleNamespace(load_dataset=fake_load_dataset)
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        from src.data.stream import hf_stream

        result = list(hf_stream("lex_glue", split="train"))

        assert result == fake_rows
        assert captured_kwargs == {
            "dataset_id": "lex_glue",
            "split": "train",
            "streaming": True,
        }

    def test_hf_stream_default_split_is_train(self, monkeypatch):
        """When split not specified, defaults to 'train'."""
        captured = {}

        def fake_load_dataset(dataset_id, split, streaming):
            captured["split"] = split
            return iter([])

        monkeypatch.setitem(
            sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
        )

        from src.data.stream import hf_stream

        list(hf_stream("foo/bar"))

        assert captured["split"] == "train"


# ---------------------------------------------------------------------------
# iter_batches
# ---------------------------------------------------------------------------


class TestIterBatches:

    def test_iter_batches_chunks_dataframe(self):
        """2500 rows + batch_size=1000 → 3 DataFrames sized (1000, 1000, 500)."""
        from src.data.stream import iter_batches

        df = pd.DataFrame({"x": range(2500)})
        batches = list(iter_batches(df, batch_size=1000))

        assert len(batches) == 3
        assert [len(b) for b in batches] == [1000, 1000, 500]
        assert all(isinstance(b, pd.DataFrame) for b in batches)
        # Round-trip: concat equals the original
        assert pd.concat(batches, ignore_index=True)["x"].tolist() == list(range(2500))

    def test_iter_batches_chunks_iterator_of_dicts(self):
        """Iterator of dicts is batched into DataFrames of the correct size."""
        from src.data.stream import iter_batches

        rows = ({"i": i, "v": i * 2} for i in range(7))
        batches = list(iter_batches(rows, batch_size=3))

        assert [len(b) for b in batches] == [3, 3, 1]
        assert batches[0]["i"].tolist() == [0, 1, 2]
        assert batches[-1]["v"].tolist() == [12]

    def test_iter_batches_empty_source_yields_nothing(self):
        """Empty input produces no batches."""
        from src.data.stream import iter_batches

        assert list(iter_batches(iter([]), batch_size=10)) == []
        assert list(iter_batches(pd.DataFrame(), batch_size=10)) == []

    def test_iter_batches_exact_multiple(self):
        """When row count is an exact multiple of batch_size, no final partial batch."""
        from src.data.stream import iter_batches

        df = pd.DataFrame({"x": range(10)})
        batches = list(iter_batches(df, batch_size=5))

        assert [len(b) for b in batches] == [5, 5]


# ---------------------------------------------------------------------------
# Network integration — opt-in, requires real Kaggle credentials
# ---------------------------------------------------------------------------


class TestKaggleCachedNetwork:
    """Real Kaggle fetch against a tiny stable dataset.

    Skipped by default (marker ``network`` is excluded via pytest addopts).
    Run explicitly with::

        KAGGLE_USERNAME=... KAGGLE_KEY=... uv run pytest -m network tests/test_streaming.py

    or export credentials via ``~/.kaggle/kaggle.json``.

    Rationale for this test: every other streaming test mocks ``kagglehub``,
    so they can't catch an upstream API break. This single network-backed
    test is the canary — if Kaggle changes its download interface or the
    dataset slug is renamed, this fails while the mocked suite stays green.
    """

    @pytest.mark.network
    def test_kaggle_cached_real_tiny_dataset(self):
        import os
        from pathlib import Path as _Path

        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        has_file = (_Path.home() / ".kaggle" / "kaggle.json").is_file()
        if not (has_env or has_file):
            pytest.skip(
                "No Kaggle credentials (KAGGLE_USERNAME+KAGGLE_KEY or "
                "~/.kaggle/kaggle.json)."
            )

        from src.data.stream import kaggle_cached

        # uciml/iris: classic iris dataset, tiny (<10 KB), stable for years
        path = kaggle_cached("uciml/iris")

        # kaggle_cached returns a pathlib.Path to the cached dataset dir
        assert isinstance(path, _Path)
        assert path.is_dir()

        # Cache must live OUTSIDE the repo (under ~/.cache/ by kagglehub default)
        project_root = _Path(__file__).resolve().parent.parent
        assert not str(path).startswith(str(project_root)), (
            f"Kaggle data leaked into the repo: {path}"
        )

        # Iris is tiny — verify the cached tree is modest
        total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        assert total_bytes <= 200_000, (
            f"uciml/iris cached as {total_bytes} bytes — upstream may have grown"
        )
