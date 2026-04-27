import proteus

from gpu_runtime import create_runtime
from test_support import (
    CudaArrayInterface,
    DataPtr,
    expect_raises,
    get_gpu_target,
    get_mlir_gpu_source,
)


def main():
    target = get_gpu_target()
    if target is None:
        return

    source = get_mlir_gpu_source()
    expect_raises(
        ValueError,
        lambda: proteus.compile(source, frontend="mlir", target=target, compiler="nvcc"),
        "MLIR frontend does not support compiler='nvcc'",
    )
    expect_raises(
        ValueError,
        lambda: proteus.compile(source, frontend="mlir", target=target, extra_args=["-O3"]),
        "MLIR frontend does not support extra_args",
    )

    runtime = create_runtime()
    device_value = runtime.malloc_i32()
    mod = proteus.compile(source, frontend="mlir", target=target, verify=True)
    kernel = mod.get_kernel("write42", [proteus.ptr])

    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=1, args=[]),
        "kernel argument count does not match argtypes",
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=(1, 1, 1, 1), args=[device_value]),
        "block must contain between 1 and 3 dimensions",
    )

    runtime.copy_h2d_i32(device_value, 0)
    kernel.launch(grid=1, block=1, args=[DataPtr(device_value)])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 42

    runtime.copy_h2d_i32(device_value, 0)
    kernel.launch(grid=(1, 1, 1), block=1, args=[CudaArrayInterface(device_value)])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 42

    runtime.free(device_value)

    print("python_gpu_mlir_validation: ok")


if __name__ == "__main__":
    main()
