import proteus

from gpu_runtime import create_runtime
from test_support import (
    CudaArrayInterface,
    DataPtr,
    EmptyCudaArrayInterface,
    expect_raises,
    get_cpp_gpu_source,
)


def main():
    target, source = get_cpp_gpu_source()
    if target is None:
        return

    runtime = create_runtime()
    device_value = runtime.malloc_i32()
    runtime.copy_h2d_i32(device_value, 0)

    mod = proteus.compile(source, frontend="cpp", target=target)
    kernel = mod.get_kernel("write_int", [proteus.ptr, proteus.i32])

    kernel.launch(grid=(1, 1, 1), block=1, args=[DataPtr(device_value), 8])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 8

    kernel.launch(grid=1, block=1, args=[CudaArrayInterface(device_value), 9])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 9

    kernel.launch(grid=[1], block=[1], args=[int(device_value), 10])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 10

    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=1, args=[object(), 1]),
        "pointer argument must be an int, None, data_ptr() object, or __cuda_array_interface__ object",
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=1, args=[EmptyCudaArrayInterface(), 1]),
        "__cuda_array_interface__ data tuple is empty",
    )

    runtime.free(device_value)

    print("python_gpu_cpp_pointer_validation: ok")


if __name__ == "__main__":
    main()
