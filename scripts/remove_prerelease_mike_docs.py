#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


_PRERELEASE_PATTERN = re.compile(r"(?i)(?:^|[^A-Za-z])(a|b|rc)\d*(?:$|[^A-Za-z])")


def is_prerelease_version(version: str) -> bool:
    return bool(_PRERELEASE_PATTERN.search(version))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove prerelease mike docs versions from a built site directory."
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        required=True,
        help="Path to the gh-pages site checkout containing versions.json.",
    )
    return parser.parse_args()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def cleanup_prerelease_docs(site_dir: Path) -> list[str]:
    versions_path = site_dir / "versions.json"
    if not versions_path.is_file():
        raise SystemExit(f"missing mike versions metadata: {versions_path}")

    versions = json.loads(versions_path.read_text(encoding="utf-8"))
    if not isinstance(versions, list):
        raise SystemExit(f"unexpected mike versions metadata shape in {versions_path}")

    kept_versions: list[dict[str, Any]] = []
    removed_versions: list[str] = []
    changed = False

    for raw_entry in versions:
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"unexpected mike version entry in {versions_path}: {raw_entry!r}")

        version = raw_entry.get("version")
        if not isinstance(version, str):
            raise SystemExit(f"missing string version in mike entry: {raw_entry!r}")

        aliases = raw_entry.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise SystemExit(f"unexpected aliases in mike entry: {raw_entry!r}")

        if is_prerelease_version(version):
            removed_versions.append(version)
            changed = True
            continue

        filtered_aliases = [alias for alias in aliases if alias != "prerelease"]
        updated_entry = dict(raw_entry)
        if filtered_aliases != aliases:
            updated_entry["aliases"] = filtered_aliases
            changed = True

        kept_versions.append(updated_entry)

    for version in removed_versions:
        _remove_path(site_dir / version)
    _remove_path(site_dir / "prerelease")

    if changed:
        versions_path.write_text(json.dumps(kept_versions, indent=2) + "\n", encoding="utf-8")

    return removed_versions


def main() -> int:
    args = _parse_args()
    removed_versions = cleanup_prerelease_docs(args.site_dir)
    if removed_versions:
        print(f"removed prerelease docs: {', '.join(removed_versions)}")
    else:
        print("removed prerelease docs: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
