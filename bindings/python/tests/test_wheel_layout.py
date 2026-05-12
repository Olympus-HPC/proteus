import pathlib
import json

import proteus
from proteus import _backend


def main():
    package_origin = proteus.__spec__.origin
    assert package_origin is not None
    package_dir = pathlib.Path(package_origin).resolve().parent
    backend_module = _backend.load_backend_module()
    backend_package_dir = pathlib.Path(backend_module.__file__).resolve().parent
    native = _backend.load_native_module()
    extension_path = pathlib.Path(native.__file__).resolve()
    assert backend_package_dir != package_dir
    assert backend_package_dir.name in {
        "proteus_backend_host",
        "proteus_backend_cuda",
        "proteus_backend_rocm",
    }
    assert extension_path.parent == backend_package_dir

    proteus_libs = list(backend_package_dir.glob("libproteus*.dylib"))
    proteus_libs.extend(backend_package_dir.glob("libproteus*.so*"))

    assert proteus_libs, f"missing vendored libproteus next to backend package: {backend_package_dir}"
    manifest = backend_package_dir / "manifest.json"
    assert manifest.exists(), f"missing active backend manifest: {manifest}"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["id"] == proteus.active_backend_variant, data
    assert data["kind"] == proteus.active_backend, data

    print("test_wheel_layout: ok")


if __name__ == "__main__":
    main()
