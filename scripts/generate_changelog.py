#!/usr/bin/env python3
"""
Generate CHANGELOG.md from Conventional Commit messages.

Reads git log since the last tag (or from the beginning for first release),
groups commits by type (feat, fix, etc.), and writes/updates CHANGELOG.md.

Usage
-----
::

    # Preview changelog for the next version
    python scripts/generate_changelog.py --repo owner/repo --current-version 1.0.0

    # Generate and write to CHANGELOG.md
    python scripts/generate_changelog.py --repo owner/repo --current-version 1.0.0 --output CHANGELOG.md

    # Dry-run (don't write)
    python scripts/generate_changelog.py --repo owner/repo --current-version 1.0.0 --dry-run

Configuration
-------------
The script reads the last N tags via ``git tag --sort=-version:refname``
to determine the previous release. The changelog follows the
`Keep a Changelog <https://keepachangelog.com/>`_ format.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


# ── Conventional commit type labels ──────────────────
COMMIT_TYPES: dict[str, str] = {
    "feat": "🚀 Features",
    "fix": "🐛 Bug Fixes",
    "docs": "📖 Documentation",
    "style": "💄 Style",
    "refactor": "♻️ Refactoring",
    "perf": "⚡ Performance",
    "test": "🧪 Tests",
    "build": "📦 Build System",
    "ci": "👷 CI/CD",
    "chore": "🔧 Maintenance",
    "revert": "⏪ Reverts",
}

# Order in which sections appear in the changelog
SECTION_ORDER = [
    "feat", "fix", "perf", "refactor", "docs", "test",
    "build", "ci", "style", "chore", "revert",
]


def run_git(*args: str) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Git command failed: git {' '.join(args)}", file=sys.stderr)
        print(f"    {exc.stderr}", file=sys.stderr)
        return ""
    except FileNotFoundError:
        print("❌ Git not found. Is this a git repository?", file=sys.stderr)
        sys.exit(1)


def get_latest_tag() -> str | None:
    """Get the most recent version tag."""
    tags = run_git("tag", "--sort=-version:refname", "--merged", "HEAD")
    if not tags:
        return None
    for tag in tags.split("\n"):
        if re.match(r"^v?\d+\.\d+\.\d+", tag):
            return tag.lstrip("v")
    return None


def parse_conventional_commit(message: str) -> dict[str, Any] | None:
    """Parse a conventional commit message into type, scope, description.

    Returns
    -------
    dict or None
        ``{"type": ..., "scope": ..., "description": ..., "breaking": bool}``
    """
    # Pattern: type(scope): description
    # or: type(scope)!: description (breaking change)
    pattern = r"^(?P<type>\w+)(\((?P<scope>[^)]*)\))?(?P<breaking>!)?\s*:\s*(?P<desc>.+)$"
    match = re.match(pattern, message.strip())
    if not match:
        return None
    return {
        "type": match.group("type"),
        "scope": match.group("scope") or "",
        "description": match.group("desc").rstrip("."),
        "breaking": bool(match.group("breaking")),
    }


def get_commits_since(since_tag: str | None = None) -> list[dict[str, Any]]:
    """Get all commits since a given tag (or all commits).

    Returns a list of dicts with keys: hash, author, date, message, parsed.
    """
    if since_tag:
        log_range = f"v{since_tag}..HEAD"
    else:
        log_range = "HEAD"

    raw = run_git(
        "log", log_range,
        "--format=%H|%an|%ai|%s",
        "--no-merges",
    )
    if not raw:
        return []

    commits: list[dict[str, Any]] = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commit_hash, author, date_str, subject = parts
        parsed = parse_conventional_commit(subject)
        if parsed:
            commits.append({
                "hash": commit_hash[:7],
                "author": author,
                "date": date_str[:10],
                "message": subject,
                "parsed": parsed,
            })
    return commits


def group_commits(commits: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group commits by conventional commit type.

    Returns a dict: type → list of formatted commit lines.
    """
    groups: dict[str, list[str]] = {}
    for c in commits:
        p = c["parsed"]
        t = p["type"]
        scope = f"**{p['scope']}:** " if p["scope"] else ""
        breaking = "💥 " if p["breaking"] else ""
        line = f"- {breaking}{scope}{p['description']} ([{c['hash']}](https://github.com/OWNER/REPO/commit/{c['hash']}))"
        groups.setdefault(t, []).append(line)
    return groups


def build_changelog(
    current_version: str,
    commits: list[dict[str, Any]],
    repo: str = "owner/repo",
    previous_version: str | None = None,
) -> str:
    """Build the changelog text for the current version.

    Parameters
    ----------
    current_version : str
        Version being released (e.g. ``1.0.0``).
    commits : list[dict]
        Parsed commit list.
    repo : str
        GitHub repository in ``owner/repo`` format.
    previous_version : str, optional
        Previous version tag.

    Returns
    -------
    str
        Changelog section for the new version.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Fix URLs in commit lines
    def _fix_url(line: str) -> str:
        return line.replace("OWNER/REPO", repo)

    groups = group_commits(commits)
    breaking = [c for c in commits if c["parsed"]["breaking"]]

    lines: list[str] = []
    lines.append(f"## [{current_version}] - {now}")
    lines.append("")

    if breaking:
        lines.append("### 💥 Breaking Changes")
        for c in breaking:
            p = c["parsed"]
            lines.append(f"- {p['description']} ([{c['hash']}](https://github.com/{repo}/commit/{c['hash']}))")
        lines.append("")

    for section_type in SECTION_ORDER:
        if section_type not in groups:
            continue
        label = COMMIT_TYPES.get(section_type, section_type.capitalize())
        lines.append(f"### {label}")
        for line in groups[section_type]:
            lines.append(_fix_url(line))
        lines.append("")

    if not groups:
        lines.append("- No significant changes.")
        lines.append("")

    lines.append(f"**Full Changelog:** https://github.com/{repo}/compare/v{previous_version or ''}...v{current_version}")
    lines.append("")

    return "\n".join(lines)


def update_changelog_file(
    version: str,
    new_section: str,
    output_path: str = "CHANGELOG.md",
    dry_run: bool = False,
) -> None:
    """Prepend the new version section to CHANGELOG.md.

    Creates the file if it doesn't exist (with header).
    """
    header = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

"""

    try:
        with open(output_path) as f:
            existing = f.read()
    except FileNotFoundError:
        existing = header

    # If the version already exists, update in place
    version_pattern = rf"## \[{re.escape(version)}\]"
    if re.search(version_pattern, existing):
        # Replace existing section
        updated = re.sub(
            rf"## \[{re.escape(version)}\].*?(?=## \[|\Z)",
            new_section.strip(),
            existing,
            count=1,
            flags=re.DOTALL,
        )
    else:
        # Prepend after header
        parts = existing.split("\n", 3)
        if len(parts) >= 4 and parts[0].startswith("#"):
            # Has header — insert after header
            insert_point = existing.index("\n", existing.index("\n", existing.index("\n") + 1) + 1) + 1
            updated = existing[:insert_point] + "\n" + new_section + existing[insert_point:]
        else:
            updated = new_section + "\n" + existing

    if dry_run:
        print("=" * 60)
        print("  DRY RUN — CHANGELOG.md would be updated with:")
        print("=" * 60)
        print(new_section)
        print("=" * 60)
        return

    with open(output_path, "w") as f:
        f.write(updated)
    print(f"✅ CHANGELOG.md updated with version {version}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate CHANGELOG.md from conventional commits",
    )
    parser.add_argument(
        "--repo", default="owner/repo",
        help="GitHub repository (owner/repo format)",
    )
    parser.add_argument(
        "--current-version", required=True,
        help="Version being released (e.g. 1.0.0)",
    )
    parser.add_argument(
        "--previous-version", default=None,
        help="Previous version (auto-detected from git tags if omitted)",
    )
    parser.add_argument(
        "--output", default="CHANGELOG.md",
        help="Output file path (default: CHANGELOG.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changelog without writing",
    )
    args = parser.parse_args()

    # Auto-detect previous version
    prev = args.previous_version or get_latest_tag()
    if prev and prev == args.current_version:
        # Same version — find the one before that
        prev = None  # Will get all commits

    print(f"Generating changelog: {prev or '(initial)'} → {args.current_version}")

    commits = get_commits_since(prev)
    if not commits:
        print("No new conventional commits found since last release.")
        # Still write an empty section
        commits = []

    print(f"Found {len(commits)} conventional commit(s).")

    new_section = build_changelog(
        current_version=args.current_version,
        commits=commits,
        repo=args.repo,
        previous_version=prev,
    )

    update_changelog_file(
        version=args.current_version,
        new_section=new_section,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
