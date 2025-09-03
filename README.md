[![docs (gh-pages)](https://github.com/Olympus-HPC/proteus/actions/workflows/gh-pages-docs.yml/badge.svg)](https://github.com/Olympus-HPC/proteus/actions/workflows/gh-pages-docs.yml)
[![Build and test](https://github.com/Olympus-HPC/proteus/actions/workflows/ci-build-test.yml/badge.svg)](https://github.com/Olympus-HPC/proteus/actions/workflows/ci-build-test.yml)
![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# <img src="docs/assets/proteus-logo.png" width="128" align="middle" /> Proteus

Proteus is a programmable Just-In-Time (JIT) compiler based on LLVM.
It embeds seamlessly into existing C++ codebases and accelerates CUDA, HIP, and
host-only C/C++ applications by leveraging runtime context to achieve
optimizations beyond static compilation.

## Description
Standard ahead-of-time (AOT) compilation can only optimize a program with the
information available at build time.
Proteus goes further.

Proteus provides APIs to embed optimizing JIT compilation directly into C/C++
applications.
By leveraging runtime context—such as the actual values of
variables during execution—Proteus can **specialize** code on the fly and apply
advanced compiler optimizations that accelerate performance beyond what static
compilation allows.

You can use Proteus in three ways:

* **Code annotations** – mark regions of existing code for JIT compilation.
* **Embedded DSL API** – build JIT code at runtime using high-level constructs.
* **C++ frontend API** – provide C++ code as strings for JIT compilation and optimization.

Proteus includes both in-memory and persistent caching, ensuring that once code
has been compiled and optimized, the cost of recompilation is avoided.

Proteus consists of an LLVM pass and a runtime library that implements JIT
compilation and optimization using LLVM as a library.

* The **code annotation** interface requires compiling your application with Clang so the Proteus LLVM pass can parse annotations.
* The **DSL** and **C++ frontend** APIs don’t depend on which AOT compiler you use.

In all cases, you link your application against the Proteus runtime library—details are provided [later](#integrating-with-your-build-system).

## Building
The project uses `cmake` for building and requires an LLVM installation
(CI tests currently cover LLVM 18, 19 with CUDA versions 12.2, and AMD ROCm versions
6.2.1, 6.3.1, 6.4.1).

See the top-level `CMakeLists.txt` for the available build options.
A typical build looks like this:
```
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install_path> ..
make install
```

The `scripts` directory contains setup scripts for building on different targets
(host-only, CUDA, ROCm) used on LLNL machines.
They also server as good starting points to adapt for other environments.
Run them from the repository root:
```bash
source setup-<target>.sh
```
These scripts load environment modules (specific to LLNL systems) and create a
`build-<hostname>-<target>-<version>` directory with a
working configuration.

## Integrating with your build system

### CMake
Integration with CMake is straightforward.
Make sure the Proteus install directory is in
`CMAKE_PREFIX_PATH`, or pass it explicitly with `-Dproteus_DIR=<install_path>` during configuration.
Then, in your project's `CMakeLists.txt` add:
```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(<target>)
```

If you only need the DSL or C++ frontend APIs, you can link directly against
`proteusFrontend`.
In this case, you don’t need to compile your target with Clang:

```cmake
find_package(proteus CONFIG REQUIRED)

target_link_libraries(<target> ... proteusFrontend ...)
```

### Make
With `make`, integrating Proteus requires adding compilation and
linking flags, for example:
```bash
CXXFLAGS += -I<install_path>/include -fpass-plugin=<install_path>/lib64/libProteusPass.so

LDFLAGS += -L <install_path>/lib64 -Wl,-rpath,<install_path>/lib64 -lproteus $(llvm-config --libs) -lclang-cpp
```
If you don't use code annotations, you can omit the `-fpass-plugin` option,
since he LLVM pass is only needed for processing annotations.

## Using

To use Proteus into your application, you need to:

1. Define your JIT code and specializations by either:
   - annotating functions (or GPU kernels),
   - building JIT compile code with the DSL API,
   - providing your source code as-a-string through the CPP frontend API

2. Update your build system to link against the Proteus runtime library. If you use the code annotation interface, you must also compile with Clang and include the Proteus LLVM plugin pass.


Proteus's core optimization technique is **runtime constant folding**.
It replaces runtime values with constants during JIT compilation, which in turn
turbo-charges classical compiler optimizations such as loop unrolling,
control-flow simplification, and constant propagation.
Think of it as doing
`constexpr`—but at runtime.

Values that can be folded include function or kernel arguments, kernel launch dimensions, launch bounds, and other runtime variables.

Here are examples of a scaled vector addition (daxpy) kernel specialized through the
different interfaces:

### Code annotation interface

We add the attribute `annotate` with the `"jit"` specifier to the kernel
function `daxpy`.
We specialize it for the scaling factor argument `A` (index 1),
and the vector size argument `N` (index 2).

```cpp
__attribute__((annotate("jit", 1, 2)))
__global__
void daxpy(double A, size_t N, double *X, double *Y)
{
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gidDim.x;
  for(int i=tid; i<N; i+=stride)
    X[i] = A*X[i] + Y[i];
}
```

### C++ frontend interface

We define the source code as a string and use `std::format` (assumes C++20, but
any string substitution method works) to specialize on the runtime values of `A`
and `N`.

```cpp
#include <format>
#include <proteus/CppJitModule.hpp>

std::string Code = std::format(R"cpp(
  extern "C"
  __global__
  void daxpy(double *X, double *Y)
  {{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int stride = blockDim.x * gidDim.x;
    for(int i=tid; i<{0}; i+=stride)
      X[i] = {1}*X[i] + Y[i];
  }})cpp", N, A);

CppJitModule CJM{"cuda", Code};
auto Kernel = CJM.getKernel<void(double *, double *)>("daxpy");
Kernel.launch(
  /* GridDim */ {NumBlocks, 1, 1},
  /* BlockDim */ {ThreadsPerBlock, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  X, Y);
```

### DSL interface

Here we use the DSL API to build the kernel function using the DSL's programming
abstractions.
The runtime values for `A` and `N` are embedded directly in the generated code.

```cpp
#include <proteus/JitFrontend.hpp>

auto createJitKernel(double A, size_t N) {
  // Targeting cuda, other targets are hip, host.
  auto J = std::make_unique<JitModule>("cuda");

  // Add the kernel with signature: void add_vectors(double *X, double *Y)
  auto KernelHandle = J->addKernel<double *, double *>("daxpy");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    // Declare local variables and argument getters.
    auto &I = F.declVar<size_t>("I");
    auto &Inc = F.declVar<size_t>("Inc");
    auto [X, Y] = F.getArgs();
    auto &RunConstA = F.defRuntimeConstant(A);
    auto &RunConstN = F.defRuntimeConst(N);

    // Compute the global thread index.
    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);

    // Compute the stride (total number of threads).
    Inc = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    // Strided loop.
    F.beginFor(I, I, RunConstN, Inc);
    {
      X[I] = RunConstA*X[I] + Y[I];
    }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}
...
auto [J, KernelHandle] = createJitKernel(N);

// Configure the CUDA kernel launch parameters.
constexpr unsigned ThreadsPerBlock = 256;
unsigned NumBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

KernelHandle.launch(
  /* GridDim */ {NumBlocks, 1, 1},
  /* BlockDim */ {ThreadsPerBlock, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  X, Y);
...
```

Proteus generates a unique specialization for each distinct set of runtime
values and caches them in memory and on disk, so JIT overhead is minimized
within and across runs.

## Documentation

The [Proteus documentation](https://olympus-hpc.github.io/proteus/) has more extensive information,
including a user's guide and developer manual.

## Contributing

We welcome contributions to Proteus in the form of pull requests targeting the
`main` branch of the repo, as well as questions, feature requests, or bug reports
via issues.

## Code of Conduct

Please note that Proteus has a [Code of Conduct](CODE_OF_CONDUCT.md).
By participating in the Proteus community, you agree to abide by its rules.

## Authors
Proteus was created by Giorgis Georgakoudis, georgakoudis1@llnl.gov.

Key contributors are:
- David Beckingsale, beckingsale1@llnl.gov
- Konstantinos Parasyris, parasyris1@llnl.gov
- John Bowen, bowen36@llnl.gov
- Zane Fink, fink12@llnl.gov
- Tal Ben Nun, bennun2@llnl.gov
- Thomas Stitt, stitt4@llnl.gov

## License

Proteus is distributed under the terms of the Apache License (Version 2.0) with
LLVM Exceptions.

All new contributions must be made under the Apache-2.0 with LLVM Exceptions license.

See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

LLNL-CODE-2000857
