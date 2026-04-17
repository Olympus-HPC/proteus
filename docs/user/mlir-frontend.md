# MLIR Frontend API

The MLIR frontend API lets you provide MLIR source code as a string and compile
it through Proteus at runtime.
It is intended for users who already produce MLIR or want direct access to
Proteus's MLIR lowering path.

Unlike the annotation interface, this path does not require compiling the
application with Clang.
The application can be built with any compatible compiler because Proteus parses
and lowers the MLIR source at runtime.

If you want to provide C++ source strings instead of MLIR, see the
[C++ frontend API](cpp-frontend.md).
If you want to construct IR programmatically rather than provide source strings,
see the [DSL API](dsl.md).

## Overview

`MLIRJitModule` is constructed from a target string plus the MLIR source code:

- target `"host"`, `"cuda"`, or `"hip"`
- source text containing a top-level MLIR `module`

For host targets, retrieve entry points with `getFunction()` and execute them
with `run()`.
For CUDA and HIP targets, retrieve GPU entry points with `getKernel()` and
launch them with grid dimensions, block dimensions, dynamic shared memory size,
stream, and kernel arguments.

## Host Example

Here is a minimal host example that compiles an MLIR function and calls it from
C++:

```cpp
#include <proteus/MLIRJitModule.h>

using namespace proteus;

static constexpr const char *Code = R"mlir(
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
)mlir";

MLIRJitModule Module{"host", Code};
auto Add = Module.getFunction<int(int, int)>("add");

int Result = Add.run(40, 2);
```

The function name passed to `getFunction()` must match the MLIR symbol name.

## GPU Example

GPU modules use `gpu.module` and `gpu.func` operations.
The top-level module must be a GPU container module, and device lowering expects
exactly one top-level `gpu.module`.
The `gpu.module` symbol name is not significant.

```cpp
#include <proteus/MLIRJitModule.h>

using namespace proteus;

static constexpr const char *Code = R"mlir(
module attributes {gpu.container_module} {
  gpu.module @device_code {
    gpu.func @write42(%out: !llvm.ptr) kernel {
      %c42 = arith.constant 42 : i32
      llvm.store %c42, %out : i32, !llvm.ptr
      gpu.return
    }
  }
}
)mlir";

MLIRJitModule Module{"cuda", Code};
auto Write42 = Module.getKernel<void(int *)>("write42");

int *DeviceBuffer = ...;
Write42.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  DeviceBuffer);
```

Use target `"hip"` instead of `"cuda"` to compile the same shape of MLIR for
HIP, assuming Proteus was built with HIP support.

## Device Module Requirements

For CUDA and HIP MLIR input, Proteus lowers a single device module to device
LLVM IR.
The input must contain exactly one top-level `gpu.module`.
If no device module is present, or if multiple `gpu.module` operations are
present, compilation fails with a diagnostic.

Kernel functions should be represented as `gpu.func` operations marked
`kernel`.
Retrieve them with `getKernel()` by their `gpu.func` symbol name.

## Kernel Function Attributes

For GPU kernels, you can set supported function attributes before launching.
For example, Proteus exposes
`JitFuncAttribute::MaxDynamicSharedMemorySize`:

```cpp
auto Kernel = Module.getKernel<void(int *)>("shmem_plain");
Kernel.setFuncAttribute(JitFuncAttribute::MaxDynamicSharedMemorySize,
                        49 * 1024);
Kernel.launch({1, 1, 1}, {1, 1, 1}, 49 * 1024, nullptr, Out);
```
