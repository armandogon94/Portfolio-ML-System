"""Real-dataset acquisition. Nothing in this module generates data.

Three access paths, in decreasing order of friction:

``kaggle_competition_cached``
    Kaggle *competitions* (IEEE-CIS). Needs a free account **and** a one-click
    acceptance of the competition rules on the web. ``kagglehub.dataset_download``
    cannot fetch a competition, which is why this function exists.

``kaggle_dataset_cached``
    Kaggle *datasets* (LendingClub, credit-card attrition). Needs a free account,
    no rules gate.

``openml_cached``
    OpenML. **No account at all.** The ULB credit-card fraud set (id 1597) lives
    here, so a reviewer with zero Kaggle presence can still reproduce a real-data
    fraud result end to end.

Everything lands in ``~/.cache/kagglehub`` / scikit-learn's OpenML cache,
outside the repository and outside the Docker build context, so ``data/raw/``
stays small.

Heavy imports happen inside functions so this module imports cheaply and mocks
cleanly in tests.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

COMPETITION_RULES_URL = "https://www.kaggle.com/competitions/{slug}/rules"


class DatasetAccessError(RuntimeError):
    """A dataset could not be fetched, with an actionable remediation attached."""


def _require_credentials(slug: str, *, kind: str) -> None:
    """Resolve Kaggle credentials into the environment or raise with instructions."""
    from src.data.kaggle_credentials import ensure_kaggle_env

    try:
        ensure_kaggle_env()
    except RuntimeError as exc:
        raise DatasetAccessError(
            f"Cannot download {kind} {slug!r}: {exc}\n\n"
            "Remediation (one time):\n"
            "  1. https://www.kaggle.com/settings/account -> 'Create New Token'\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json\n"
            "No synthetic fallback is provided on purpose; see "
            "docs/adr/0003-real-data-over-synthetic.md."
        ) from exc


def kaggle_dataset_cached(slug: str, *, filename: str | None = None) -> Path:
    """Download a Kaggle *dataset* to the local kagglehub cache.

    Args:
        slug: Dataset slug, e.g. ``"sakshigoyal7/credit-card-customers"``.
        filename: Optional file inside the dataset. When given the returned path
            points at that file; otherwise at the dataset directory.

    Returns:
        Path to the cached dataset directory or file.

    Raises:
        DatasetAccessError: Credentials missing, or the download failed.
    """
    cached = _cached_kaggle_dataset(slug, filename=filename)
    if cached is not None:
        logger.info("Using cached Kaggle dataset %s at %s", slug, cached)
        return cached

    _require_credentials(slug, kind="dataset")

    import kagglehub

    try:
        cache_dir = Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # noqa: BLE001 - re-raised with remediation
        raise DatasetAccessError(
            f"kagglehub could not download dataset {slug!r}: {exc}\n"
            f"Check the slug at https://www.kaggle.com/datasets/{slug}"
        ) from exc

    logger.info("Kaggle dataset %s cached at %s", slug, cache_dir)
    return _resolve_within(cache_dir, filename)


def _cached_kaggle_dataset(slug: str, *, filename: str | None) -> Path | None:
    """Return the newest complete kagglehub cache version without a network call."""
    cache_root = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))
    versions = cache_root / "datasets" / slug / "versions"
    if not versions.is_dir():
        return None

    def version_key(path: Path) -> tuple[int, str]:
        return (int(path.name), path.name) if path.name.isdigit() else (-1, path.name)

    for version in sorted(
        (path for path in versions.iterdir() if path.is_dir()),
        key=version_key,
        reverse=True,
    ):
        if filename is None:
            if any(path.is_file() for path in version.rglob("*")):
                return version
            continue
        matches = sorted(version.rglob(filename))
        if matches:
            return matches[0]
    return None


def kaggle_competition_cached(slug: str, *, filename: str | None = None) -> Path:
    """Download a Kaggle *competition* dataset to the local kagglehub cache.

    Competitions are gated behind an explicit acceptance of the competition
    rules that cannot be scripted. When that acceptance is missing, Kaggle
    returns a 403 and this function converts it into the exact URL to visit.

    Args:
        slug: Competition slug, e.g. ``"ieee-fraud-detection"``.
        filename: Optional file inside the competition bundle.

    Returns:
        Path to the cached competition directory or file.

    Raises:
        DatasetAccessError: Credentials missing, rules not accepted, or the
            download failed.
    """
    _require_credentials(slug, kind="competition")

    import kagglehub

    if not hasattr(kagglehub, "competition_download"):
        raise DatasetAccessError(
            "The installed kagglehub has no competition_download(). "
            "Upgrade with: uv add 'kagglehub>=0.3.4'"
        )

    try:
        cache_dir = Path(kagglehub.competition_download(slug))
    except Exception as exc:  # noqa: BLE001 - re-raised with remediation
        raise DatasetAccessError(
            f"kagglehub could not download competition {slug!r}: {exc}\n\n"
            "The most common cause is that the competition rules have not been "
            "accepted on this Kaggle account. That is a one-click action and it "
            "cannot be automated:\n"
            f"  {COMPETITION_RULES_URL.format(slug=slug)}\n"
            "  -> 'I Understand and Accept'\n"
            "If the rules are already accepted and ~/.kaggle/access_token is the "
            "only credential, install a classic API token at "
            "~/.kaggle/kaggle.json; Kaggle competition downloads do not accept "
            "the OAuth token on this machine.\n"
            "Then re-run this command."
        ) from exc

    logger.info("Kaggle competition %s cached at %s", slug, cache_dir)
    return _resolve_within(cache_dir, filename)


def _restore_openml_row_id(
    frame: pd.DataFrame,
    details: dict[str, Any],
    *,
    data_home: str | Path,
) -> pd.DataFrame:
    """Restore a documented OpenML row id that scikit-learn omits from ``frame``."""
    import pandas as pd

    row_id = details.get("row_id_attribute")
    if not isinstance(row_id, str) or not row_id or row_id in frame.columns:
        return frame

    source_url = details.get("url")
    if not isinstance(source_url, str):
        raise RuntimeError(f"OpenML documents row id {row_id!r} but provides no source URL.")
    parsed = urlparse(source_url)
    host = parsed.netloc.removeprefix("www.")
    cached = Path(data_home) / "openml" / host / parsed.path.lstrip("/")
    candidates = (cached, Path(f"{cached}.gz"))
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError(
            f"OpenML documents row id {row_id!r}, but its cached ARFF was not found at "
            f"{candidates}. Refusing to discard the temporal split key."
        )

    attributes: list[str] = []
    data_line = 0
    if source.suffix == ".gz":
        handle = gzip.open(source, "rt", encoding="utf-8")
    else:
        handle = source.open("rt", encoding="utf-8")
    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            match = re.match(
                r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
                stripped,
                flags=re.IGNORECASE,
            )
            if match:
                attributes.append(next(group for group in match.groups() if group is not None))
            if stripped.casefold() == "@data":
                data_line = line_number
                break
    if row_id not in attributes or data_line == 0:
        raise RuntimeError(f"{source} does not contain documented OpenML row id {row_id!r}.")

    row_ids = pd.read_csv(
        source,
        compression="gzip" if source.suffix == ".gz" else None,
        header=None,
        names=attributes,
        skiprows=data_line,
        usecols=[row_id],
        quotechar="'",
    )[row_id]
    if len(row_ids) != len(frame) or row_ids.isna().any():
        raise RuntimeError(
            f"OpenML row id {row_id!r} has {len(row_ids)} usable rows; "
            f"the parsed frame has {len(frame)}."
        )

    restored = frame.copy()
    restored.insert(0, row_id, pd.to_numeric(row_ids, errors="raise").astype("float32"))
    return restored


def openml_cached(data_id: int):
    """Fetch an OpenML dataset as a pandas DataFrame. Requires no credentials.

    Args:
        data_id: OpenML dataset id, e.g. ``1597`` for ULB credit-card fraud.

    Returns:
        ``pandas.DataFrame`` with the features and the target column appended.
    """
    from sklearn.datasets import fetch_openml, get_data_home

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = _restore_openml_row_id(bunch.frame, bunch.details, data_home=get_data_home())
    logger.info("OpenML dataset %s loaded: %d rows x %d cols", data_id, *frame.shape)
    return frame


def _resolve_within(directory: Path, filename: str | None) -> Path:
    """Return ``directory / filename`` if asked, validating that it exists."""
    if filename is None:
        return directory

    candidate = directory / filename
    if candidate.exists():
        return candidate

    # kagglehub occasionally nests one level deeper than the slug implies.
    matches = sorted(directory.rglob(filename))
    if matches:
        return matches[0]

    available = sorted(p.name for p in directory.rglob("*") if p.is_file())[:20]
    raise DatasetAccessError(
        f"{filename!r} not found under {directory}. Files present: {available}"
    )


def sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, streamed so large files stay off-heap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
