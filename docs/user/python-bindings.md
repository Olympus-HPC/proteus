# Python Bindings

The Python API compiles C++, LLVM IR, or MLIR source with `proteus.compile()`.
After compilation,
retrieve a native entry point by supplying its exact ABI signature.

## Host Functions

Build a signature by calling the return-type descriptor with the argument-type
descriptors:

```python
import proteus

source = r'''
extern "C" int plus1(int x) {
  return x + 1;
}
'''

mod = proteus.compile(source, frontend="cpp", target="host")
plus1 = mod.get_function(
    "plus1",
    signature=proteus.i32(proteus.i32),
)

assert plus1(41) == 42
```

The `signature` parameter is keyword-only.
A function with no arguments uses an empty descriptor call,
such as `proteus.i32()`.
Host functions may return any supported scalar or pointer descriptor,
or `proteus.void`.

## GPU Kernels

CUDA and HIP kernel signatures must return `proteus.void`:

```python
kernel = mod.get_kernel(
    "write_int",
    signature=proteus.void(proteus.ptr, proteus.i32),
)

kernel.launch(
    grid=1,
    block=256,
    args=[device_output, 42],
)
```

`proteus.void` is valid only as a return descriptor.
It cannot appear in the argument list.

## Supported ABI Descriptors

Signatures preserve the exact width and signedness of every descriptor:

| Descriptor | Native ABI type |
| --- | --- |
| `proteus.i8` | signed 8-bit integer |
| `proteus.i32` | signed 32-bit integer |
| `proteus.i64` | signed 64-bit integer |
| `proteus.u32` | unsigned 32-bit integer |
| `proteus.u64` | unsigned 64-bit integer |
| `proteus.f32` | 32-bit floating point |
| `proteus.f64` | 64-bit floating point |
| `proteus.ptr` | opaque pointer |
| `proteus.void` | return only; no value |

A constructed `proteus.Signature` exposes read-only `restype` and `argtypes`
properties.
`argtypes` is always a tuple.
For example,
`repr(proteus.f64(proteus.f64, proteus.f64))` is
`proteus.f64(proteus.f64, proteus.f64)`.

Proteus does not infer native types from Python builtins,
function annotations,
decorators,
or runtime argument values.
Python types such as `int` and `float`,
and `ctypes` types,
are not valid signature descriptors.

!!! warning

    Proteus cannot verify that a supplied signature matches the compiled
    native symbol.
    A mismatch in return type,
    argument order,
    width,
    signedness,
    or pointer position can cross the native ABI incorrectly and cause
    corrupted results or a process failure.
