# Python Bindings

The Python API compiles C++, LLVM IR, or MLIR source with `proteus.compile()`.
After compilation,
retrieve a native entry point by supplying its exact ABI signature.

For workflows that need to inspect or transform LLVM IR before compilation,
Proteus also provides an owned staged module under `proteus.llvm`.
The staged API is separate from the executable `proteus.Module` returned by
`proteus.compile()`.

## Staged LLVM Modules

Create staged modules from LLVM IR text or bitcode bytes:

```python
import proteus

module = proteus.llvm.Module.from_ir(llvm_ir)
module = proteus.llvm.Module.from_bitcode(bitcode)

bitcode = module.to_bitcode()
llvm_ir = module.to_ir()
module.verify()
```

`from_bitcode()` accepts `bytes`,
and `to_bitcode()` returns `bytes`.
The bitcode boundary preserves the module target triple,
data layout,
attributes,
and metadata without exposing an LLVM context or `LLVMModuleRef`.
Use ordinary Python file operations when reading or writing bitcode files.

`clone()` returns an independent module.
`Module.link()` likewise copies its inputs into a new context,
leaving every input module usable:

```python
linked = proteus.llvm.Module.link([module_a, module_b])
working = linked.clone()
```

The following preparation operations mutate the module and return it,
which allows either step-by-step or fluent use:

```python
module.prune()
module.internalize(preserve=["kernel_name"])
```

`prune()` removes Proteus compiler bookkeeping and,
by default,
unsets `externally_initialized` on globals.
Pass `unset_externally_initialized=False` to retain those flags.
`internalize()` preserves the named global symbols and internalizes the rest.

## Replay Specialization

CUDA and HIP backend builds expose the runtime transformations used by
Proteus kernel specialization:

```python
module.specialize_launch_dimensions(
    grid=(16, 1, 1),
    block=(256, 1, 1),
)
module.assume_launch_dimension_ranges(
    grid=(16, 1, 1),
    block=(256, 1, 1),
)
module.set_launch_bounds(
    "kernel_name",
    max_threads_per_block=256,
    min_blocks_per_sm=2,
)
```

Exact launch-dimension specialization replaces recognized dimension queries.
Range specialization attaches `[0, dimension)` information to recognized
thread and block index calls.
All dimensions must be positive 32-bit unsigned values.

Argument specialization accepts one sized buffer containing the native
`void*[]` kernel-argument array:

```python
module.specialize_arguments(
    "kernel_name",
    arguments=argument_pointer_array,
    indexes=[0, 3],
)
```

The buffer must be one-dimensional,
C-contiguous,
and contain native pointer-sized entries.
Its length must match the LLVM function arity.
Each selected entry points to the storage for that argument value.
Proteus reads selected values synchronously and retains neither the buffer nor
its pointers after the method returns.

## Optimization and Object Emission

`proteus.Config.current()` exposes the effective process configuration without
allowing Python to mutate Proteus global state:

```python
base = proteus.Config.current().codegen("kernel_name")
config = base.replace(
    method="serial",
    pipeline="default<O3>",
    codegen_opt_level=3,
)

module.optimize("gfx942", config=config)
object_bytes = module.emit_object("gfx942", config=config)
```

Calling `codegen()` without a kernel name returns the global code-generation
configuration.
With a name,
Proteus applies its normal per-kernel tuned-configuration lookup and global
fallback.
`replace()` returns an independent immutable value;
it does not change environment-derived or tuned process configuration.

The configuration exposes `method`,
`pipeline`,
`opt_level`,
`codegen_opt_level`,
the runtime-specialization policy flags,
and tuned launch-bound values.
Codegen methods are normalized to lowercase `rtc`,
`serial`,
or `parallel`.

`optimize()` mutates the staged module.
`emit_object()` returns device-object `bytes` and performs backend codegen on
an internal clone,
so the optimized module remains unchanged.
CUDA object emission supports `rtc`;
HIP supports the methods available in the active Proteus build.
CPU-only builds support parsing,
serialization,
linking,
preparation,
and generic optimization,
but report device specialization and object emission as unsupported.

An external transform can safely participate through a bitcode round trip:

```python
mneme_module = mneme.parse_bitcode(proteus_module.to_bitcode())
remove_auto_initialize(mneme_module)
proteus_module = proteus.llvm.Module.from_bitcode(
    mneme_module.to_bitcode()
)
```

Both libraries must use compatible LLVM bitcode versions.
An incompatible or malformed stream raises a parse error from
`from_bitcode()`.

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
