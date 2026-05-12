#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_DEFAULT_PROJECTS = [
    "proteus-python-backend-host-llvm22",
    "proteus-python-backend-cuda12-llvm22",
    "proteus-python-backend-rocm72",
]
_REPOSITORY = "Olympus-HPC/proteus"
_API_URL = "https://api.github.com"


def _normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_prefix(project_name: str) -> str:
    return project_name.replace("-", "_") + "-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static PEP 503 simple index for Proteus backend wheels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the channel subdirectories should be written.",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Backend distribution name to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", _REPOSITORY),
        help="GitHub repository in OWNER/REPO form.",
    )
    parser.add_argument(
        "--api-url",
        default=_API_URL,
        help="Base URL for the GitHub API.",
    )
    parser.add_argument(
        "--releases-json",
        type=Path,
        help="Optional JSON file containing the GitHub releases payload.",
    )
    return parser.parse_args()


def _read_releases_from_file(path: Path) -> list[dict[str, Any]]:
    if str(path) == "-":
        return json.load(sys.stdin)
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_releases(api_url: str, repository: str) -> list[dict[str, Any]]:
    releases: list[dict[str, Any]] = []
    token = os.environ.get("GITHUB_TOKEN")
    page = 1
    while True:
        url = f"{api_url}/repos/{repository}/releases?per_page=100&page={page}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        with urllib.request.urlopen(request) as response:
            page_releases = json.load(response)
        if not page_releases:
            break
        releases.extend(page_releases)
        if len(page_releases) < 100:
            break
        page += 1
    return releases


def _release_asset_url(release: dict[str, Any], asset: dict[str, Any]) -> str:
    browser_url = asset.get("browser_download_url")
    if browser_url:
        return str(browser_url)

    tag_name = release["tag_name"]
    repository = str(release.get("html_url") or "").removesuffix(f"/releases/tag/{tag_name}")
    if repository:
        return f"{repository}/releases/download/{tag_name}/{asset['name']}"
    raise ValueError(
        f"release asset {asset['name']} for tag {tag_name} is missing browser_download_url"
    )


def _bucket_release_assets(
    releases: list[dict[str, Any]], project_names: list[str]
) -> dict[str, dict[str, list[dict[str, str]]]]:
    prefixes = {project: _wheel_prefix(project) for project in project_names}
    channels: dict[str, dict[str, list[dict[str, str]]]] = {
        "simple": {project: [] for project in project_names},
        "test": {project: [] for project in project_names},
    }

    def release_sort_key(release: dict[str, Any]) -> tuple[str, str]:
        return (str(release.get("published_at") or ""), str(release.get("tag_name") or ""))

    for release in sorted(releases, key=release_sort_key, reverse=True):
        if release.get("draft"):
            continue
        channel = "test" if release.get("prerelease") else "simple"
        for asset in release.get("assets", []):
            name = str(asset.get("name") or "")
            if not name.endswith(".whl"):
                continue
            for project, prefix in prefixes.items():
                if name.startswith(prefix):
                    channels[channel][project].append(
                        {
                            "filename": name,
                            "href": _release_asset_url(release, asset),
                        }
                    )
                    break

    for project_entries in channels.values():
        for entries in project_entries.values():
            entries.sort(key=lambda entry: entry["filename"], reverse=True)
    return channels


def _render_root_page(project_entries: dict[str, list[dict[str, str]]]) -> str:
    anchors = []
    for project_name, entries in project_entries.items():
        if not entries:
            continue
        normalized = _normalize_project_name(project_name)
        anchors.append(
            f'    <a href="{normalized}/">{html.escape(project_name)}</a>'
        )

    body = "\n".join(anchors)
    return f"""<!DOCTYPE html>
<html>
  <body>
{body}
  </body>
</html>
"""


def _render_project_page(project_name: str, entries: list[dict[str, str]]) -> str:
    anchors = "\n".join(
        f'    <a href="{html.escape(entry["href"], quote=True)}">{html.escape(entry["filename"])}</a>'
        for entry in entries
    )
    return f"""<!DOCTYPE html>
<html>
  <body>
    <h1>{html.escape(project_name)}</h1>
{anchors}
  </body>
</html>
"""


def _write_channel(output_dir: Path, project_entries: dict[str, list[dict[str, str]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        _render_root_page(project_entries), encoding="utf-8"
    )

    for project_name, entries in project_entries.items():
        if not entries:
            continue
        project_dir = output_dir / _normalize_project_name(project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "index.html").write_text(
            _render_project_page(project_name, entries), encoding="utf-8"
        )


def main() -> int:
    args = _parse_args()
    project_names = args.project or list(_DEFAULT_PROJECTS)

    try:
        releases = (
            _read_releases_from_file(args.releases_json)
            if args.releases_json
            else _fetch_releases(args.api_url, args.repository)
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"failed to load GitHub releases: {exc}") from exc

    channels = _bucket_release_assets(releases, project_names)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    _write_channel(args.output_dir / "simple", channels["simple"])
    _write_channel(args.output_dir / "test", channels["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
