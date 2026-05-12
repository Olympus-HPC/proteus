import importlib
import importlib.util
import pathlib
import sys
import types


def main():
    init_path = pathlib.Path(__file__).resolve().parents[3] / "python" / "proteus" / "__init__.py"
    package_name = "proteus_init_under_test"

    backend_module = types.ModuleType(f"{package_name}._backend")
    backend_module.available_backends = lambda: ["host"]
    backend_module.available_backend_variants = lambda: ["host_llvm22"]
    backend_module.active_backend = lambda: "host"
    backend_module.active_backend_variant = lambda: "host_llvm22"

    native_module = types.ModuleType(f"{package_name}._proteus")
    native_module.__doc__ = "fake native module"
    native_module.__file__ = "/tmp/fake_proteus_backend.so"
    native_module.compile = lambda *args, **kwargs: None
    native_module.i32 = object()
    native_module.has_cuda = False
    native_module.has_hip = False

    sys.modules[backend_module.__name__] = backend_module
    sys.modules[native_module.__name__] = native_module

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

    exported = importlib.import_module(package_name)
    namespace = {}
    exec(f"from {package_name} import *", namespace, namespace)

    assert namespace["active_backend"] == "host"
    assert namespace["active_backend_variant"] == "host_llvm22"
    assert "compile" in namespace
    assert "i32" in namespace
    assert exported.__doc__ == "fake native module"
    assert exported.__file__ == "/tmp/fake_proteus_backend.so"

    print("test_proteus_init_exports: ok")


if __name__ == "__main__":
    main()
