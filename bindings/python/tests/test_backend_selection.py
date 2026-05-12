import importlib.util
import os
import pathlib
import sys
import types
from unittest import mock


class FakeEntryPoint:
    def __init__(self, getter):
        self._getter = getter

    def load(self):
        return self._getter


def make_backend_module(name, kind, compatible_env, default_variant, variants):
    module = types.ModuleType(name)
    variant_by_id = {item["id"]: item for item in variants}

    def available_variant_specs():
        return list(variants)

    def active_variant_spec():
        explicit = os.environ.get("PROTEUS_BACKEND_VARIANT", "").strip()
        if explicit:
            if explicit not in variant_by_id:
                raise ImportError(explicit)
            return variant_by_id[explicit]
        return variant_by_id[default_variant]

    def is_runtime_compatible():
        return os.environ.get(compatible_env) == "1"

    def load_native_module():
        variant = active_variant_spec()
        return types.SimpleNamespace(__name__=variant["id"], marker=variant["id"])

    def get_backend():
        return {"kind": kind, "module": name, "priority": {"host": 10, "rocm": 20, "cuda": 30}[kind]}

    module.available_variant_specs = available_variant_specs
    module.active_variant_spec = active_variant_spec
    module.is_runtime_compatible = is_runtime_compatible
    module.load_native_module = load_native_module
    module.get_backend = get_backend
    return module


def main():
    loader_path = (
        pathlib.Path(__file__).resolve().parents[3] / "python" / "proteus" / "_backend.py"
    )
    spec = importlib.util.spec_from_file_location("proteus_backend_selector_under_test", loader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    fake_modules = {
        "fake_backend_host": make_backend_module(
            "fake_backend_host",
            "host",
            "FAKE_HOST",
            "host_llvm22",
            [
                {"id": "host_llvm20", "kind": "host"},
                {"id": "host_llvm22", "kind": "host"},
            ],
        ),
        "fake_backend_cuda": make_backend_module(
            "fake_backend_cuda",
            "cuda",
            "FAKE_CUDA",
            "cuda12_llvm22",
            [
                {"id": "cuda12_llvm20", "kind": "cuda"},
                {"id": "cuda12_llvm22", "kind": "cuda"},
            ],
        ),
        "fake_backend_rocm": make_backend_module(
            "fake_backend_rocm",
            "rocm",
            "FAKE_ROCM",
            "rocm72",
            [
                {"id": "rocm64", "kind": "rocm"},
                {"id": "rocm72", "kind": "rocm"},
            ],
        ),
    }
    sys.modules.update(fake_modules)

    eps = [FakeEntryPoint(fake_modules[name].get_backend) for name in fake_modules]
    with mock.patch.object(module, "_entry_points_for_group", return_value=eps):
        assert module.available_backends() == ["cuda", "rocm", "host"]
        assert module.available_backend_variants() == [
            "cuda12_llvm20",
            "cuda12_llvm22",
            "rocm64",
            "rocm72",
            "host_llvm20",
            "host_llvm22",
        ]

        with mock.patch.dict(os.environ, {}, clear=True):
            assert module.active_backend() == "host"
            assert module.active_backend_variant() == "host_llvm22"

        with mock.patch.dict(os.environ, {"FAKE_CUDA": "1"}, clear=True):
            assert module.active_backend() == "cuda"
            assert module.active_backend_variant() == "cuda12_llvm22"

        with mock.patch.dict(
            os.environ,
            {"PROTEUS_BACKEND_KIND": "rocm", "FAKE_ROCM": "1"},
            clear=True,
        ):
            assert module.active_backend() == "rocm"
            assert module.active_backend_variant() == "rocm72"

        with mock.patch.dict(
            os.environ,
            {"PROTEUS_BACKEND_VARIANT": "host_llvm20"},
            clear=True,
        ):
            assert module.active_backend() == "host"
            assert module.active_backend_variant() == "host_llvm20"

        with mock.patch.dict(os.environ, {}, clear=True):
            native = module.load_native_module()
            assert native.marker == "host_llvm22"

        with mock.patch.dict(
            os.environ,
            {"PROTEUS_BACKEND_VARIANT": "cuda12_llvm22", "FAKE_CUDA": "1"},
            clear=True,
        ):
            try:
                module.load_native_module()
            except RuntimeError as exc:
                assert "locked" in str(exc), str(exc)
            else:
                raise AssertionError("expected backend process lock failure")

    print("test_backend_selection: ok")


if __name__ == "__main__":
    main()
