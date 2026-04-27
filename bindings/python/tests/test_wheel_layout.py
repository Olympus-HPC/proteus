import pathlib

import proteus
import proteus._proteus as native

def main():
    package_dir = pathlib.Path(proteus.__file__).resolve().parent
    extension_path = pathlib.Path(native.__file__).resolve()
    assert extension_path.parent == package_dir

    proteus_libs = list(package_dir.glob("libproteus*.dylib"))
    proteus_libs.extend(package_dir.glob("libproteus*.so*"))

    assert proteus_libs, f"missing vendored libproteus next to package: {package_dir}"

    print("test_wheel_layout: ok")


if __name__ == "__main__":
    main()
