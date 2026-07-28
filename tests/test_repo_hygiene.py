"""Repository-wide typography guard.

The house style forbids U+2014 (the em dash) anywhere in the tree, prose and code
alike. This module enforces that over **every tracked file**, enumerated with
``git ls-files -z``, not over a hand-written glob.

That scope is the whole point. An earlier sweep checked ``git ls-files '*.md'``,
reported zero, and was accurate about markdown while 103 tracked source files still
carried the character. A gate whose walk cannot reach the files that break it is a
gate that cannot fail, so :func:`test_the_scan_reaches_beyond_documentation` asserts
the walk itself.

Two forms are checked:

* the literal glyph, matched on the raw UTF-8 bytes;
* the backslash-u escape of the same code point, which a grep for the glyph cannot
  see and which reaches a reader as a real em dash the moment JSON or Python decodes
  it. :data:`ESCAPED_EM_DASH` is assembled from two string literals so that naming
  the pattern does not plant it.

Binary files are excluded by git's own heuristic, a NUL byte in the first 8000, so a
PNG that happens to contain the byte sequence is not reported as prose.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Built from its code point so this file can enforce the rule without breaking it.
EM_DASH = chr(0x2014)
EM_DASH_BYTES = EM_DASH.encode("utf-8")

#: The escaped spelling, assembled the same way and for the same reason.
ESCAPED_EM_DASH = "\\" + "u2014"
ESCAPED_EM_DASH_BYTES = ESCAPED_EM_DASH.encode("ascii")

#: git treats a file as binary when a NUL byte appears in this many leading bytes.
_BINARY_SNIFF_BYTES = 8000

_FIX = (
    "Rewrite each one with punctuation that fits the sentence: a colon, a comma, a "
    "semicolon, parentheses, or a restructure. Do not substitute a hyphen."
)


def _tracked_paths() -> list[Path]:
    """Every file git tracks, as absolute paths.

    Skips rather than passes when git is unavailable. A vacuous pass here would
    recreate exactly the failure mode this module exists to prevent.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - git is present in CI and locally
        pytest.skip(f"git is not runnable, so tracked files cannot be enumerated: {exc}")
    if result.returncode != 0:  # pragma: no cover - only outside a checkout
        pytest.skip("not a git checkout, so the tracked-file list cannot be enumerated")
    names = result.stdout.decode("utf-8").split("\0")
    return [REPO_ROOT / name for name in names if name]


def _text_payloads() -> list[tuple[Path, bytes]]:
    """Tracked files that git would treat as text, with their raw bytes."""
    payloads = []
    for path in _tracked_paths():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:_BINARY_SNIFF_BYTES]:
            continue
        payloads.append((path, data))
    return payloads


def _hits(needle: bytes) -> list[str]:
    """``path:count`` for every tracked text file containing ``needle``."""
    return [
        f"{path.relative_to(REPO_ROOT)}: {data.count(needle)}"
        for path, data in _text_payloads()
        if needle in data
    ]


def test_no_tracked_text_file_contains_an_em_dash():
    offenders = _hits(EM_DASH_BYTES)
    assert not offenders, (
        f"{len(offenders)} tracked text file(s) contain U+2014.\n"
        + _FIX
        + "\n"
        + "\n".join(offenders)
    )


def test_no_tracked_text_file_contains_an_escaped_em_dash():
    """The escape survives a glyph grep and decodes back into the character."""
    offenders = _hits(ESCAPED_EM_DASH_BYTES)
    assert not offenders, (
        f"{len(offenders)} tracked text file(s) contain the {ESCAPED_EM_DASH} escape.\n"
        + _FIX
        + "\n"
        + "\n".join(offenders)
    )


def test_the_scan_reaches_beyond_documentation():
    """Guard the guard: narrowing the walk back to markdown must break a test.

    The earlier markdown-only sweep was truthful and useless at the same time. These
    assertions pin the walk to the file kinds that actually carried the character:
    Python, TypeScript, YAML, Dockerfiles, shell, and dotfiles at the repository root.
    """
    scanned = {path.relative_to(REPO_ROOT) for path, _ in _text_payloads()}
    assert scanned, "the tracked-file walk found nothing at all"

    suffixes = {path.suffix for path in scanned}
    for required in (".py", ".ts", ".tsx", ".yaml", ".yml", ".sh", ".toml"):
        assert required in suffixes, f"the walk never reached a {required} file"

    names = {path.name for path in scanned}
    assert ".gitignore" in names, "the walk never reached a dotfile"
    assert any(path.name.endswith(".Dockerfile") for path in scanned), (
        "the walk never reached a Dockerfile"
    )

    markdown = {path for path in scanned if path.suffix == ".md"}
    assert len(scanned) > 5 * len(markdown), (
        f"only {len(scanned)} files scanned against {len(markdown)} markdown files; "
        "the walk has been narrowed back toward documentation"
    )
