from importlib import import_module


def _native_module():
    return import_module(f"{__name__}._proteus")


def __getattr__(name: str):
    return getattr(_native_module(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native_module())))


def get_backend():
    raise RuntimeError("Backend metadata must be provided by a concrete backend package")
