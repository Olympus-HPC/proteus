from importlib import import_module


def _native_module():
    return import_module("proteus_backend_cu12._proteus")


def __getattr__(name: str):
    return getattr(_native_module(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native_module())))


def get_backend():
    return {
        "id": "cuda12",
        "module": "proteus_backend_cu12",
        "native_module": "proteus_backend_cu12._proteus",
        "priority": 20,
    }
