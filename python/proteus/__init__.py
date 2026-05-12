from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_output

from ._backend import available_backend_variants, available_backends

_METADATA_EXPORTS = ["available_backends", "available_backend_variants"]


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


def _backend_loader():
    from . import _backend

    return _backend


def _native_module():
    return import_module(f"{__name__}._proteus")


def _export_native_api() -> list[str]:
    if not available_backends():
        return []

    native = _native_module()
    globals()["__doc__"] = native.__doc__
    globals()["__file__"] = getattr(native, "__file__", __file__)

    exports: list[str] = []
    for name in dir(native):
        if name.startswith("__") and name not in {"__doc__", "__file__"}:
            continue
        globals()[name] = getattr(native, name)
        exports.append(name)
    return exports


try:
    __version__ = _dist_version("proteus-python")
except PackageNotFoundError:
    __version__ = _fallback_version()


_HAS_BACKENDS = bool(available_backends())
__all__ = sorted(
    set(_METADATA_EXPORTS)
    | set(_export_native_api())
    | ({"active_backend", "active_backend_variant"} if _HAS_BACKENDS else set())
)


def __getattr__(name: str):
    if name == "active_backend":
        return _backend_loader().active_backend()
    if name == "active_backend_variant":
        return _backend_loader().active_backend_variant()
    return getattr(_native_module(), name)


def __dir__():
    exported = set(globals()) | {"active_backend", "active_backend_variant"}
    try:
        exported.update(dir(_native_module()))
    except ImportError:
        pass
    return sorted(exported)
