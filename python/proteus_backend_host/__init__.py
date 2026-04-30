from importlib import import_module


def _native_module():
    return import_module("proteus_backend_host._proteus")


def __getattr__(name: str):
    return getattr(_native_module(), name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native_module())))


def get_backend():
    return {
        "id": "host",
        "module": "proteus_backend_host",
        "native_module": "proteus_backend_host._proteus",
        "priority": 10,
    }
