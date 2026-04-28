# LLVM IR Frontend API

The LLVM IR frontend API lets you provide LLVM IR directly as text or bitcode
and compile it through Proteus at runtime.
It is intended for users who already produce LLVM IR and want Proteus to handle
target selection, caching, and execution without going through the C++ or MLIR
frontends.

Unlike the annotation interface, this path does not require compiling the
application with Clang.
The application can be built with any compatible compiler because Proteus parses
the LLVM IR at runtime.

If you want to provide C++ source strings instead of LLVM IR, see the
[C++ frontend API](cpp-frontend.md).
If you want direct access to Proteus's MLIR lowering path, see the
[MLIR frontend API](mlir-frontend.md).
If you want to construct IR programmatically rather than provide source text,
see the [DSL API](dsl.md).

## Overview

`LLVMIRJitModule` is constructed from a target string plus LLVM IR input:

- target `"host"`, `"cuda"`, or `"hip"`
- LLVM IR source text or LLVM bitcode bytes
- optional `LLVMIRInputKind`, which defaults to `LLVMIRInputKind::Auto`

`LLVMIRInputKind` controls how the input is parsed:

- `Auto`: try bitcode first, then fall back to text IR
- `TextIR`: require textual LLVM IR
- `Bitcode`: require LLVM bitcode

For host targets, retrieve entry points with `getFunction()` and execute them
with `run()`.
For CUDA and HIP targets, retrieve device entry points with `getKernel()` and
launch them with grid dimensions, block dimensions, dynamic shared memory size,
stream, and kernel arguments.

If the input module omits the target triple or data layout, Proteus fills them
in for the selected target before compilation.

## Host Example

Here is a minimal host example that compiles an LLVM IR function and calls it
from C++:

```cpp
#include <proteus/LLVMIRJitModule.h>

using namespace proteus;

static constexpr const char *Code = R"llvm(
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
)llvm";

LLVMIRJitModule Module{"host", Code};
auto Add = Module.getFunction<int(int, int)>("add");

int Result = Add.run(40, 2);
```

The symbol name passed to `getFunction()` must match the LLVM function symbol.

## GPU Example

GPU modules use regular LLVM IR with a device target triple and a kernel entry
point that is valid for the selected backend.

CUDA example:

```cpp
#include <proteus/LLVMIRJitModule.h>

using namespace proteus;

static constexpr const char *Code = R"llvm(
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @write42(ptr addrspace(1) %out) {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @write42, !"kernel", i32 1}
)llvm";

LLVMIRJitModule Module{"cuda", Code};
auto Write42 = Module.getKernel<void(int *)>("write42");

int *DeviceBuffer = ...;
Write42.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  DeviceBuffer);
```

Use target `"hip"` instead of `"cuda"` to compile for HIP, assuming Proteus
was built with HIP support.
The input must already be LLVM IR that is valid for HIP execution, such as IR
with an AMDGPU target triple and compatible kernel conventions.

## Python Bindings

The Python bindings expose the same frontend through
`proteus.compile(..., frontend="llvmir")`.

```python
import proteus

source = r"""
define i32 @plus1(i32 %x) {
entry:
  %sum = add i32 %x, 1
  ret i32 %sum
}
"""

mod = proteus.compile(source, frontend="llvmir", target="host")
plus1 = mod.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])
assert plus1(41) == 42
```

In the current Python API, LLVM IR input is provided as text through a string
or a path to a `.ll` file.
