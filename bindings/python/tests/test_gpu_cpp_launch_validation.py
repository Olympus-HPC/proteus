import proteus

from gpu_runtime import create_runtime
from test_support import expect_raises, get_cpp_gpu_source


def main():
    target, source = get_cpp_gpu_source()
    if target is None:
        return

    runtime = create_runtime()
    device_value = runtime.malloc_i32()

    mod = proteus.compile(source, frontend="cpp", target=target)
    kernel = mod.get_kernel("write_int", [proteus.ptr, proteus.i32])

    expect_raises(
        ValueError,
        lambda: mod.get_function(
            "write_int", restype=None, argtypes=[proteus.ptr, proteus.i32]
        ),
        "Target is a GPU model, cannot directly run functions, use launch()",
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=1, args=[device_value]),
        "kernel argument count does not match argtypes",
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=object(), block=1, args=[device_value, 1]),
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=(1, 1, 1, 1), block=1, args=[device_value, 1]),
        "grid must contain between 1 and 3 dimensions",
    )
    expect_raises(
        TypeError,
        lambda: kernel.launch(grid=1, block=(), args=[device_value, 1]),
        "block must contain between 1 and 3 dimensions",
    )

    runtime.free(device_value)

    print("python_gpu_cpp_launch_validation: ok")


if __name__ == "__main__":
    main()
