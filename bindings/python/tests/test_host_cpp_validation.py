import ctypes

import proteus

from test_support import CudaArrayInterface, expect_raises


def main():
    assert repr(proteus.i8) == "proteus.i8"
    assert repr(proteus.i32) == "proteus.i32"
    assert repr(proteus.i64) == "proteus.i64"
    assert repr(proteus.u32) == "proteus.u32"
    assert repr(proteus.u64) == "proteus.u64"
    assert repr(proteus.f32) == "proteus.f32"
    assert repr(proteus.f64) == "proteus.f64"
    assert repr(proteus.ptr) == "proteus.ptr"
    assert isinstance(proteus.has_cuda, bool)
    assert isinstance(proteus.has_hip, bool)
    assert isinstance(proteus.has_mlir, bool)

    source = 'extern "C" int forty_two() { return 42; }'
    expect_raises(
        ValueError,
        lambda: proteus.compile(source, frontend="nope", target="host"),
        "frontend must be 'cpp' or 'mlir'",
    )
    expect_raises(
        ValueError,
        lambda: proteus.compile(
            source, frontend="cpp", target="host", compiler="not-a-compiler"
        ),
        "compiler must be 'clang' or 'nvcc'",
    )
    expect_raises(
        ValueError,
        lambda: proteus.compile(
            "module {}", frontend="mlir", target="host", compiler="nvcc"
        ),
        "MLIR frontend does not support compiler='nvcc'",
    )
    expect_raises(
        ValueError,
        lambda: proteus.compile(
            "module {}", frontend="mlir", target="host", extra_args=["-O3"]
        ),
        "MLIR frontend does not support extra_args",
    )
    plus1 = proteus.compile(
        r'''
extern "C" int plus1(int x) { return x + 1; }
extern "C" int load0(const int *xs) { return xs[0]; }
extern "C" void store0(int *xs, int value) { xs[0] = value; }
''',
        frontend="cpp",
        target="host",
    )
    plus1_fn = plus1.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])
    load0 = plus1.get_function("load0", restype=proteus.i32, argtypes=[proteus.ptr])
    store0 = plus1.get_function(
        "store0", restype=None, argtypes=[proteus.ptr, proteus.i32]
    )
    assert repr(plus1_fn) == "<proteus.Function name='plus1' restype=proteus.i32 argtypes=[proteus.i32]>"
    assert plus1_fn(41) == 42

    values = (ctypes.c_int * 2)(11, 22)
    assert load0(values) == 11
    store0(values, 33)
    assert values[0] == 33

    scalar = ctypes.c_int(44)
    assert load0(ctypes.pointer(scalar)) == 44

    class ArrayInterface:
        def __init__(self, ptr):
            self.__array_interface__ = {
                "shape": (1,),
                "typestr": "<i4",
                "data": (ptr, False),
                "version": 3,
            }

    assert load0(ArrayInterface(ctypes.addressof(scalar))) == 44
    expect_raises(
        TypeError,
        lambda: load0(CudaArrayInterface(ctypes.addressof(scalar))),
        "pointer argument must be an int, None, data_ptr() object, __array_interface__ object, or ctypes object",
    )
    expect_raises(
        TypeError,
        lambda: load0(object()),
        "pointer argument must be an int, None, data_ptr() object, __array_interface__ object, or ctypes object",
    )

    print("python_host_cpp_validation: ok")


if __name__ == "__main__":
    main()
