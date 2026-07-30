"""Repository-wide typography guard.

The house style forbids U+2014 (the em dash) anywhere in the tree, prose and code
alike. This module enforces that over every tracked file and every untracked,
nonignored file, enumerated with ``git ls-files``, not over a hand-written glob.

That scope is the whole point. An earlier sweep checked ``git ls-files '*.md'``,
reported zero, and was accurate about markdown while 103 tracked source files still
carried the character. A gate whose walk cannot reach the files that break it is a
gate that cannot fail, so :func:`test_the_scan_reaches_beyond_documentation` asserts
the walk itself.

Two kinds of spelling are checked:

* the literal glyph, matched on the raw UTF-8 bytes;
* every *encoded* spelling that decodes back to the same code point, listed in
  :data:`ENCODED_SPELLINGS`. A grep for the glyph cannot see any of them, and each one
  reaches a reader as a real em dash the moment JSON, Python or an HTML parser decodes
  it. This tree is Python, TypeScript, TSX, YAML and SVG, so the backslash escapes and
  the HTML entities are both live re-entry routes: the named entity in a TSX component
  or in a generated SVG diagram renders as the character itself.

Every needle is assembled from two string literals, so naming a pattern here does not
plant it in the tree and trip this module against itself.

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

#: Encoded spellings that decode back to U+2014, as ``(spelling, fold_case)``.
#:
#: Each is assembled from two literals for the same reason as :data:`EM_DASH`. The
#: backslash escapes are case-significant, lowercase ``u`` and uppercase ``U`` being
#: different escapes rather than variants of one, so both are listed and matched as
#: written. HTML entity names and the hexadecimal marker are case-insensitive per the
#: HTML spec, so those are matched against a folded haystack instead.
ENCODED_SPELLINGS: tuple[tuple[str, bool], ...] = (
    ("\\" + "u2014", False),  # JSON, JavaScript and Python, four hex digits
    ("\\" + "U00002014", False),  # Python and C, eight hex digits
    ("\\" + "N{EM DASH}", False),  # Python, by Unicode character name
    ("&" + "mdash;", True),  # HTML, XML, SVG and JSX, by entity name
    ("&#" + "8212;", True),  # HTML numeric entity, decimal
    ("&#" + "x2014;", True),  # HTML numeric entity, hexadecimal
)

#: git treats a file as binary when a NUL byte appears in this many leading bytes.
_BINARY_SNIFF_BYTES = 8000

_FIX = (
    "Rewrite each one with punctuation that fits the sentence: a colon, a comma, a "
    "semicolon, parentheses, or a restructure. Do not substitute a hyphen."
)


def _repository_paths() -> list[Path]:
    """Every tracked or untracked nonignored file, as absolute paths.

    Skips rather than passes when git is unavailable. A vacuous pass here would
    recreate exactly the failure mode this module exists to prevent.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - git is present in CI and locally
        pytest.skip(f"git is not runnable, so repository files cannot be enumerated: {exc}")
    if result.returncode != 0:  # pragma: no cover - only outside a checkout
        pytest.skip("not a git checkout, so the tracked-file list cannot be enumerated")
    names = result.stdout.decode("utf-8").split("\0")
    return [REPO_ROOT / name for name in names if name]


def test_repository_walk_includes_untracked_nonignored_files(monkeypatch):
    """A new public file must not sit outside the typography gate."""
    listing = b"tracked.py\0new_report.md\0"

    class Completed:
        returncode = 0
        stdout = listing

    def fake_run(command, **kwargs):
        assert "--others" in command
        assert "--exclude-standard" in command
        return Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _repository_paths() == [
        REPO_ROOT / "tracked.py",
        REPO_ROOT / "new_report.md",
    ]


def _text_payloads() -> list[tuple[Path, bytes]]:
    """Public repository candidates that git would treat as text, with raw bytes."""
    payloads = []
    for path in _repository_paths():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:_BINARY_SNIFF_BYTES]:
            continue
        payloads.append((path, data))
    return payloads


def _hits(needle: bytes, fold_case: bool = False) -> list[str]:
    """``path:count`` for every tracked text file containing ``needle``.

    ``fold_case`` lowercases the haystack, for spellings whose grammar is
    case-insensitive; the needle is already lowercase in that case.
    """
    hits = []
    for path, data in _text_payloads():
        haystack = data.lower() if fold_case else data
        count = haystack.count(needle)
        if count:
            hits.append(f"{path.relative_to(REPO_ROOT)}: {count}")
    return hits


def test_no_tracked_text_file_contains_an_em_dash():
    offenders = _hits(EM_DASH_BYTES)
    assert not offenders, (
        f"{len(offenders)} tracked text file(s) contain U+2014.\n"
        + _FIX
        + "\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(("spelling", "fold_case"), ENCODED_SPELLINGS, ids=lambda v: str(v))
def test_no_tracked_text_file_contains_an_encoded_em_dash(spelling: str, fold_case: bool):
    """Each encoded spelling survives a glyph grep and decodes back to the character."""
    needle = spelling.lower() if fold_case else spelling
    offenders = _hits(needle.encode("ascii"), fold_case=fold_case)
    assert not offenders, (
        f"{len(offenders)} tracked text file(s) contain the {spelling} spelling of "
        f"U+2014.\n" + _FIX + "\n" + "\n".join(offenders)
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


def test_every_known_encoding_route_is_still_covered():
    """Guard the guard, second axis: emptying the spelling table must break a test.

    An empty ``parametrize`` argument list collects one case and reports it SKIPPED,
    not failed, so the encoded-spelling test can be disarmed without a red run. That
    is the same defect class as a walk that cannot reach the offending files. This
    pins the routes that actually decode to the character in this tree: the backslash
    escapes for Python, JSON and TypeScript, and the three HTML entity forms for TSX
    and the generated SVG diagrams.
    """
    spellings = {spelling for spelling, _ in ENCODED_SPELLINGS}
    for required in ("\\" + "u2014", "\\" + "U00002014", "&" + "mdash;"):
        assert required in spellings, f"the {required} spelling is no longer checked"

    entity_forms = {spelling for spelling in spellings if spelling.startswith("&")}
    assert len(entity_forms) == 3, (
        f"expected the named, decimal and hexadecimal entity forms, found {entity_forms}"
    )
    assert all(fold for spelling, fold in ENCODED_SPELLINGS if spelling.startswith("&")), (
        "HTML entity names are case-insensitive, so they must be matched case-folded"
    )
