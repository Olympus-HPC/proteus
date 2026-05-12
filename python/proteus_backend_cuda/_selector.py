from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path

_PACKAGE = "proteus_backend_cuda"
_BACKEND_KIND = "cuda"
_BACKEND_PRIORITY = 30
_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def _manifest() -> dict[str, object]:
    data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    data.setdefault("kind", _BACKEND_KIND)
    data.setdefault("module", f"{_PACKAGE}._proteus")
    return data


def available_variant_specs() -> list[dict[str, object]]:
    return [_manifest()]


def available_variants() -> list[str]:
    return [str(_manifest()["id"])]


def active_variant_spec() -> dict[str, object]:
    explicit_variant = os.environ.get("PROTEUS_BACKEND_VARIANT", "").strip()
    variant = _manifest()
    if explicit_variant and explicit_variant != str(variant["id"]):
        raise ImportError(
            f"Requested Proteus CUDA backend variant {explicit_variant!r} is not available. "
            f"Available variants: {available_variants()}"
        )
    return variant


def active_variant() -> str:
    return str(active_variant_spec()["id"])


def is_runtime_compatible() -> bool:
    for env_name in ("PROTEUS_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env_name):
            return True
    return Path("/usr/local/cuda").exists()


def load_native_module():
    return import_module(str(_manifest()["module"]))


def get_backend():
    return {
        "kind": _BACKEND_KIND,
        "module": _PACKAGE,
        "priority": _BACKEND_PRIORITY,
    }
