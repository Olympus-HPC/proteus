import pathlib
import tempfile

import proteus

from gpu_runtime import create_runtime
from test_support import get_gpu_target, get_mlir_gpu_source


def main():
    target = get_gpu_target()
    if target is None:
        return

    source = get_mlir_gpu_source()
    runtime = create_runtime()
    device_value = runtime.malloc_i32()
    runtime.copy_h2d_i32(device_value, 0)

    mod = proteus.compile(source, frontend="mlir", target=target, verify=True)
    assert mod.get_function_address("write42") != 0
    kernel = mod.get_kernel("write42", [proteus.ptr])
    kernel.launch(grid=1, block=1, args=[device_value])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 42

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "kernel.mlir"
        path.write_text(source)
        file_mod = proteus.compile(path, frontend="mlir", target=target, verify=True)
        file_kernel = file_mod.get_kernel("write42", [proteus.ptr])
        runtime.copy_h2d_i32(device_value, 0)
        file_kernel.launch(grid=[1], block=(1, 1, 1), args=[int(device_value)])
        runtime.sync()
        assert runtime.copy_d2h_i32(device_value) == 42

    runtime.free(device_value)

    print("python_gpu_mlir_smoke: ok")


if __name__ == "__main__":
    main()
