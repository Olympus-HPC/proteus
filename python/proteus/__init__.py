from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_output

from . import _proteus
from ._proteus import *  # noqa: F401,F403

__doc__ = _proteus.__doc__


def _fallback_version():
    repo_root = Path(__file__).resolve().parents[2]
    try:
        sha = check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=DEVNULL,
            text=True,
        ).strip()
    except (CalledProcessError, FileNotFoundError):
        return "dev"

    return f"g{sha}" if sha else "dev"


try:
    __version__ = _dist_version("proteus-python")
except PackageNotFoundError:
    __version__ = _fallback_version()
