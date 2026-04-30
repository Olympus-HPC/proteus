import pathlib

import proteus
import proteus._proteus as native

def main():
    package_dir = pathlib.Path(proteus.__file__).resolve().parent
    extension_path = pathlib.Path(native.__file__).resolve()
    native_package_dir = extension_path.parent
    assert native_package_dir != package_dir
    assert native_package_dir.name in {"proteus_backend_host", "proteus_backend_cu12"}

    proteus_libs = list(native_package_dir.glob("libproteus*.dylib"))
    proteus_libs.extend(native_package_dir.glob("libproteus*.so*"))

    assert proteus_libs, f"missing vendored libproteus next to backend package: {native_package_dir}"

    print("test_wheel_layout: ok")


if __name__ == "__main__":
    main()
