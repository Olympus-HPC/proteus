import ctypes

import proteus

from test_support import CudaArrayInterface, expect_raises


def main():
    assert repr(proteus.void) == "proteus.void"
    assert repr(proteus.i8) == "proteus.i8"
    assert repr(proteus.i32) == "proteus.i32"
    assert repr(proteus.i64) == "proteus.i64"
    assert repr(proteus.u32) == "proteus.u32"
    assert repr(proteus.u64) == "proteus.u64"
    assert repr(proteus.f32) == "proteus.f32"
    assert repr(proteus.f64) == "proteus.f64"
    assert repr(proteus.ptr) == "proteus.ptr"

    arg_descriptors = (
        proteus.i8,
        proteus.i32,
        proteus.i64,
        proteus.u32,
        proteus.u64,
        proteus.f32,
        proteus.f64,
        proteus.ptr,
    )
    for descriptor in (proteus.void, *arg_descriptors):
        assert isinstance(descriptor, proteus.Type)
        assert isinstance(descriptor(), proteus.Signature)

    signature = proteus.f64(proteus.f64, proteus.f64)
    assert repr(signature.restype) == "proteus.f64"
    assert tuple(map(repr, signature.argtypes)) == (
        "proteus.f64",
        "proteus.f64",
    )
    assert isinstance(signature.argtypes, tuple)
    assert repr(signature) == "proteus.f64(proteus.f64, proteus.f64)"

    zero_arg_signature = proteus.i32()
    assert repr(zero_arg_signature.restype) == "proteus.i32"
    assert zero_arg_signature.argtypes == ()
    assert repr(zero_arg_signature) == "proteus.i32()"

    all_arg_signature = proteus.void(*arg_descriptors)
    assert repr(all_arg_signature.restype) == "proteus.void"
    assert tuple(map(repr, all_arg_signature.argtypes)) == tuple(
        map(repr, arg_descriptors)
    )

    expect_raises(TypeError, lambda: proteus.Signature())
    expect_raises(
        AttributeError,
        lambda: setattr(signature, "restype", proteus.i32),
    )
    expect_raises(
        AttributeError,
        lambda: setattr(signature, "argtypes", (proteus.i32,)),
    )

    invalid_argtypes = (
        int,
        float,
        ctypes.c_int32,
        ctypes.c_int32(),
        object(),
        1,
        None,
    )
    for invalid in invalid_argtypes:
        expect_raises(
            TypeError,
            lambda invalid=invalid: proteus.i32(invalid),
            "signature arguments must be Proteus type descriptors",
        )
    expect_raises(
        TypeError,
        lambda: proteus.i32(proteus.void),
        "proteus.void is only valid as a signature return type",
    )

    assert isinstance(proteus.has_cuda, bool)
    assert isinstance(proteus.has_hip, bool)
    assert isinstance(proteus.has_mlir, bool)

    source = 'extern "C" int forty_two() { return 42; }'
    expect_raises(
        ValueError,
        lambda: proteus.compile(source, frontend="nope", target="host"),
        "frontend must be 'cpp', 'llvmir', or 'mlir'",
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
    mod = proteus.compile(
        r'''
#include <cstdint>

extern "C" std::int32_t forty_two() { return 42; }
extern "C" std::int32_t plus1(std::int32_t x) { return x + 1; }
extern "C" std::int8_t echo_i8(std::int8_t x) { return x; }
extern "C" std::int32_t echo_i32(std::int32_t x) { return x; }
extern "C" std::int64_t echo_i64(std::int64_t x) { return x; }
extern "C" std::uint32_t echo_u32(std::uint32_t x) { return x; }
extern "C" std::uint64_t echo_u64(std::uint64_t x) { return x; }
extern "C" float echo_f32(float x) { return x; }
extern "C" double add_f64(double x, double y) { return x + y; }
extern "C" void *echo_ptr(void *x) { return x; }
extern "C" std::int32_t load0(const std::int32_t *xs) { return xs[0]; }
extern "C" void store0(std::int32_t *xs, std::int32_t value) { xs[0] = value; }
''',
        frontend="cpp",
        target="host",
    )

    expect_raises(
        TypeError,
        lambda: mod.get_function("plus1", proteus.i32(proteus.i32)),
    )
    expect_raises(
        TypeError,
        lambda: mod.get_function(
            "plus1", restype=proteus.i32, argtypes=[proteus.i32]
        ),
    )
    expect_raises(
        TypeError,
        lambda: mod.get_function("plus1", signature=int),
    )
    expect_raises(
        TypeError,
        lambda: mod.get_kernel(
            "missing_kernel", signature=proteus.i32(proteus.i32)
        ),
        "kernel signatures must return proteus.void",
    )
    expect_raises(
        TypeError,
        lambda: mod.get_kernel("plus1", proteus.void(proteus.i32)),
    )
    expect_raises(
        TypeError,
        lambda: mod.get_kernel("plus1", argtypes=[proteus.i32]),
    )

    forty_two = mod.get_function("forty_two", signature=proteus.i32())
    plus1_fn = mod.get_function(
        "plus1", signature=proteus.i32(proteus.i32)
    )
    echo_i8 = mod.get_function("echo_i8", signature=proteus.i8(proteus.i8))
    echo_i32 = mod.get_function(
        "echo_i32", signature=proteus.i32(proteus.i32)
    )
    echo_i64 = mod.get_function(
        "echo_i64", signature=proteus.i64(proteus.i64)
    )
    echo_u32 = mod.get_function(
        "echo_u32", signature=proteus.u32(proteus.u32)
    )
    echo_u64 = mod.get_function(
        "echo_u64", signature=proteus.u64(proteus.u64)
    )
    echo_f32 = mod.get_function(
        "echo_f32", signature=proteus.f32(proteus.f32)
    )
    add_f64 = mod.get_function(
        "add_f64", signature=proteus.f64(proteus.f64, proteus.f64)
    )
    echo_ptr = mod.get_function(
        "echo_ptr", signature=proteus.ptr(proteus.ptr)
    )
    load0 = mod.get_function("load0", signature=proteus.i32(proteus.ptr))
    store0 = mod.get_function(
        "store0", signature=proteus.void(proteus.ptr, proteus.i32)
    )
    assert (
        repr(plus1_fn)
        == "<proteus.Function name='plus1' signature=proteus.i32(proteus.i32)>"
    )
    assert forty_two() == 42
    assert plus1_fn(41) == 42
    expect_raises(
        TypeError,
        lambda: plus1_fn(),
        "function argument count does not match signature",
    )

    assert echo_i8(-(2**7)) == -(2**7)
    assert echo_i32(-(2**31)) == -(2**31)
    assert echo_i64(-(2**63)) == -(2**63)
    assert echo_u32(2**32 - 1) == 2**32 - 1
    assert echo_u64(2**64 - 1) == 2**64 - 1
    assert echo_f32(1.5) == 1.5
    assert add_f64(1.25, 2.5) == 3.75

    values = (ctypes.c_int * 2)(11, 22)
    assert echo_ptr(values) == ctypes.addressof(values)
    assert echo_ptr(None) is None
    assert load0(values) == 11
    assert store0(values, 33) is None
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
