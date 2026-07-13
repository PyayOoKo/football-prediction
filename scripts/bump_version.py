#!/usr/bin/env python3
"""
Bump the project version following Semantic Versioning (semver).

Updates ``pyproject.toml`` and optionally ``__init__.py``.

Usage
-----
::

    # Bump patch: 1.0.0 → 1.0.1
    python scripts/bump_version.py --current 1.0.0 --part patch

    # Bump minor: 1.0.0 → 1.1.0
    python scripts/bump_version.py --current 1.0.0 --part minor

    # Bump major: 1.0.0 → 2.0.0
    python scripts/bump_version.py --current 1.0.0 --part major

    # Bump to a specific version
    python scripts/bump_version.py --current 1.0.0 --set-version 2.0.0-beta.1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_version(version: str) -> tuple[int, int, int, str]:
    """Parse a semver string into (major, minor, patch, pre-release).

    Parameters
    ----------
    version : str
        Version string like ``1.0.0`` or ``1.0.0-beta.1``.

    Returns
    -------
    tuple[int, int, int, str]
        ``(major, minor, patch, pre_release)`` where pre_release is
        ``""`` for stable versions.
    """
    match = re.match(
        r"^(\d+)\.(\d+)\.(\d+)(-([a-zA-Z0-9.]+))?(\+([a-zA-Z0-9.]+))?$",
        version,
    )
    if not match:
        raise ValueError(f"Invalid semver: '{version}'")
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        match.group(5) or "",
    )


def format_version(major: int, minor: int, patch: int, pre: str = "") -> str:
    """Format version components back into a string."""
    base = f"{major}.{minor}.{patch}"
    if pre:
        base += f"-{pre}"
    return base


def bump_version(
    current: str,
    part: str = "patch",
    set_version: str | None = None,
) -> str:
    """Compute the next version.

    Parameters
    ----------
    current : str
        Current version (e.g. ``1.0.0``).
    part : str
        Part to bump: ``major``, ``minor``, ``patch``.
    set_version : str, optional
        Explicit version to set (overrides ``part``).

    Returns
    -------
    str
        New version string.
    """
    if set_version:
        parse_version(set_version)  # validate
        return set_version

    major, minor, patch, pre = parse_version(current)

    if part == "major":
        return format_version(major + 1, 0, 0)
    elif part == "minor":
        return format_version(major, minor + 1, 0)
    elif part == "patch":
        return format_version(major, minor, patch + 1)
    else:
        raise ValueError(f"Unknown part: '{part}'. Use: major, minor, patch")


def update_file(path: Path, pattern: str, replacement: str) -> bool:
    """Update a file using regex substitution.

    Parameters
    ----------
    path : Path
        File to update.
    pattern : str
        Regex pattern.
    replacement : str
        Replacement string.

    Returns
    -------
    bool
        True if the file was modified.
    """
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return False

    original = path.read_text()
    updated = re.sub(pattern, replacement, original)
    if updated == original:
        print(f"⚠️  No version string found in {path}")
        return False

    path.write_text(updated)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump project version following SemVer",
    )
    parser.add_argument(
        "--current", required=True,
        help="Current version (e.g. 1.0.0)",
    )
    parser.add_argument(
        "--part", default="patch",
        choices=["major", "minor", "patch"],
        help="Version part to bump (default: patch)",
    )
    parser.add_argument(
        "--set-version", default=None,
        help="Explicit version (overrides --part)",
    )
    args = parser.parse_args()

    new_version = bump_version(
        current=args.current,
        part=args.part,
        set_version=args.set_version,
    )

    print(f"  {args.current} → {new_version}")

    # Update pyproject.toml
    project_root = Path(__file__).resolve().parent.parent
    pyproject = project_root / "pyproject.toml"

    updated_pyproject = update_file(
        pyproject,
        rf'version = "{re.escape(args.current)}"',
        f'version = "{new_version}"',
    )

    # Update src/__init__.py
    init_py = project_root / "src" / "__init__.py"
    updated_init = update_file(
        init_py,
        rf'__version__ = "{re.escape(args.current)}"',
        f'__version__ = "{new_version}"',
    )

    if updated_pyproject or updated_init:
        print(f"✅ Version bumped to {new_version}")
        return 0
    else:
        print("❌ No files were updated.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
