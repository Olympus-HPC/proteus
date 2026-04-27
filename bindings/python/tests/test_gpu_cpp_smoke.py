import proteus

from gpu_runtime import create_runtime
from test_support import get_cpp_gpu_source


def main():
    target, source = get_cpp_gpu_source()
    if target is None:
        return

    runtime = create_runtime()
    device_value = runtime.malloc_i32()
    runtime.copy_h2d_i32(device_value, 0)

    mod = proteus.compile(source, frontend="cpp", target=target)
    kernel = mod.get_kernel("write_int", [proteus.ptr, proteus.i32])
    assert repr(kernel) == "<proteus.Kernel name='write_int' restype=None argtypes=[proteus.ptr, proteus.i32]>"
    kernel.launch(grid=1, block=(1, 1, 1), args=[device_value, 7])
    runtime.sync()
    assert runtime.copy_d2h_i32(device_value) == 7
    runtime.free(device_value)

    print("python_gpu_cpp_smoke: ok")


if __name__ == "__main__":
    main()
