# C++ Frontend API

The C++ frontend API lets you provide source code as a string, similar to
CUDA/HIP RTC, but with a portable and higher-level interface.
It supports runtime C++ template instantiation and source-level substitution
for embedding runtime values.

Unlike the annotation interface, this path does not require compiling the
application with Clang. The application can be built with any compatible
compiler because Proteus compiles the JIT source at runtime.

If you want annotation-driven integration into existing code, see
[Code Annotations](annotations.md).
If you want to construct IR programmatically rather than compile source
strings, see the [DSL API](dsl.md).

## Overview

`CppJitModule` is constructed from a target string plus the source code:

- target `"host"`, `"cuda"`, or `"hip"`
- optional extra compiler arguments, passed as `std::vector<std::string>`
- optional compiler backend, which defaults to
  `CppJitCompilerBackend::Clang`

For CUDA modules, you can also request `CppJitCompilerBackend::Nvcc`.

## Basic Module Construction

Here is an illustrative example that compiles a DAXPY kernel through this API:

```cpp
#include <format>
#include <proteus/CppJitModule.h>

double *X = ...;
double *Y = ...;
double A = ...;
size_t N = ...;

std::string Code = std::format(R"cpp(
  extern "C"
  __global__
  void daxpy(double *X, double *Y)
  {{
    int Tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int Stride = blockDim.x * gridDim.x;
    for (int I = Tid; I < {0}; I += Stride)
      X[I] = {1} * X[I] + Y[I];
  }})cpp", N, A);

CppJitModule Module{"cuda", Code};
auto Kernel = Module.getKernel<void(double *, double *)>("daxpy");
Kernel.launch(
  /* GridDim */ {NumBlocks, 1, 1},
  /* BlockDim */ {ThreadsPerBlock, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  X, Y);
```

The code string contains placeholders `{0}` and `{1}`.
Here `std::format` substitutes the runtime values of vector size `N` and the
scaling factor `A` before compilation.
Use `extern "C"` to avoid C++ name mangling so the function or kernel can be
retrieved by name.

## Host Example

The same pattern for targeting the CPU is very similar:
instead of `getKernel()` use `getFunction()`, and execute the JIT-compiled code
with `run()` instead of `launch()`.

```cpp
std::string Code = std::format(R"cpp(
  extern "C"
  void daxpy(double *X, double *Y)
  {{
    for (int I = 0; I < {0}; ++I)
      X[I] = {1} * X[I] + Y[I];
  }})cpp", N, A);

CppJitModule Module{"host", Code};
auto Func = Module.getFunction<void(double *, double *)>("daxpy");
Func.run(X, Y);
```

## GPU Example

GPU targets use the same module abstraction.
Select `"cuda"` or `"hip"` as the target string, retrieve the entry point with
`getKernel()`, and launch it with grid dimensions, block dimensions, dynamic
shared memory size, stream, and kernel arguments.

## Template Instantiation

The C++ frontend API also supports runtime instantiation of C++ templates.
Here is a simple example:

```cpp
const char *Code = R"cpp(
  template<int V>
  __global__ void foo(int *Ret) {
    *Ret = V;
  }

  template<typename T>
  __global__ void bar(T *Ret) {
    *Ret = 42;
  }
)cpp";

int *RetFoo = ...;
double *RetBar = ...;

CppJitModule Module{"cuda", Code};
auto &InstanceFoo = Module.instantiate("foo", "3");
auto &InstanceBar = Module.instantiate("bar", "double");

InstanceFoo.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  RetFoo);

InstanceBar.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  RetBar);
```

The `Code` block defines two kernels:

- `foo` with a non-type template parameter (`int V`)
- `bar` with a type template parameter (`typename T`)

Use `Module.instantiate()` to create concrete instantiations by passing the
kernel name along with the template arguments as strings.
Repeated requests for the same instantiation reuse the cached handle inside the
module.

The returned handle can then be launched using the same `launch()` API as
before.

## Optional Compile Args and Backend Selection

You can pass extra compile arguments or select the backend explicitly:

```cpp
CppJitModule HostModule{"host", Code, {"-DMY_OFFSET=10"}};
CppJitModule DeviceModule{"cuda", Code, {}, CppJitCompilerBackend::Nvcc};
```

`CppJitCompilerBackend::Clang` is the default backend.
For CUDA modules, `CppJitCompilerBackend::Nvcc` is also available when you want
to compile through the NVCC toolchain instead.

## Kernel Function Attributes

For GPU kernels and instantiated kernels, you can set supported function
attributes before launching.
For example, Proteus exposes
`JitFuncAttribute::MaxDynamicSharedMemorySize`:

```cpp
auto Kernel = Module.getKernel<void(int *)>("shmem_plain");
Kernel.setFuncAttribute(JitFuncAttribute::MaxDynamicSharedMemorySize,
                        49 * 1024);
Kernel.launch({1, 1, 1}, {1, 1, 1}, 49 * 1024, nullptr, Out);
```
