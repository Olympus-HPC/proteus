from ._selector import (
    active_variant,
    active_variant_spec,
    available_variant_specs,
    available_variants,
    get_backend,
    is_runtime_compatible,
    load_native_module,
)


def __getattr__(name: str):
    return getattr(load_native_module(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_native_module())))
