import importlib.util
import pathlib
import sys
from unittest import mock


def main():
    loader_path = (
        pathlib.Path(__file__).resolve().parents[3] / "python" / "proteus" / "_backend.py"
    )
    spec = importlib.util.spec_from_file_location("proteus_loader_under_test", loader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with mock.patch.object(module, "_entry_points_for_group", return_value=[]):
        try:
            module.load_backend_module()
        except ImportError as exc:
            assert "No Proteus backend is installed" in str(exc), str(exc)
        else:
            raise AssertionError("expected ImportError when no backend entry points are present")

    print("test_backend_loader_zero_backends: ok")


if __name__ == "__main__":
    main()
