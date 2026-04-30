from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from typing import Iterable


_BACKEND_ENTRYPOINT_GROUP = "proteus.backends"


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    module_name: str
    native_module_name: str
    priority: int


def _entry_points_for_group(group: str):
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    return list(discovered.get(group, []))


def _normalize_backend_spec(raw_spec: object) -> BackendSpec:
    if not isinstance(raw_spec, dict):
        raise ImportError(
            "Proteus backend entry point must return a dict with backend metadata"
        )

    try:
        backend_id = str(raw_spec["id"])
        module_name = str(raw_spec["module"])
        native_module_name = str(raw_spec["native_module"])
        priority = int(raw_spec["priority"])
    except KeyError as exc:
        raise ImportError(
            f"Proteus backend metadata is missing required key {exc.args[0]!r}"
        ) from exc

    return BackendSpec(
        backend_id=backend_id,
        module_name=module_name,
        native_module_name=native_module_name,
        priority=priority,
    )


def _discover_backend_specs() -> list[BackendSpec]:
    specs = []
    for ep in _entry_points_for_group(_BACKEND_ENTRYPOINT_GROUP):
        raw_spec = ep.load()()
        specs.append(_normalize_backend_spec(raw_spec))
    return specs


def available_backends() -> list[str]:
    return [spec.backend_id for spec in _sorted_specs(_discover_backend_specs())]


def _sorted_specs(specs: Iterable[BackendSpec]) -> list[BackendSpec]:
    return sorted(specs, key=lambda spec: (spec.priority, spec.backend_id), reverse=True)


def _selected_backend_spec() -> BackendSpec:
    specs = _sorted_specs(_discover_backend_specs())
    if not specs:
        raise ImportError(
            "No Proteus backend is installed. Install proteus-python or "
            "proteus-python[cuda12]."
        )
    return specs[0]


def active_backend() -> str:
    return _selected_backend_spec().backend_id


def load_backend_module():
    return import_module(_selected_backend_spec().module_name)


def load_native_module():
    return import_module(_selected_backend_spec().native_module_name)
