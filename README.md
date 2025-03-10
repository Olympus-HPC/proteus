[![docs (gh-pages)](https://github.com/Olympus-HPC/proteus/actions/workflows/gh-pages-docs.yml/badge.svg)](https://github.com/Olympus-HPC/proteus/actions/workflows/gh-pages-docs.yml)
[![Build and test](https://github.com/Olympus-HPC/proteus/actions/workflows/ci-build-test.yml/badge.svg)](https://github.com/Olympus-HPC/proteus/actions/workflows/ci-build-test.yml)
![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# <img src="docs/assets/proteus-logo.png" width="128" align="middle" /> Proteus

Proteus optimizes C/C++ code execution, including CUDA/HIP kernels, using
Just-In-Time (JIT) compilation powered by LLVM, to apply runtime optimizations
that leverage runtime context.

## Description
Proteus JIT compilation leverages runtime context information that is
unavailable during ahead-of-time (AOT) compilation, for example, the runtime
values of variables during program execution.
Proteus *specializes* JIT-generated code on this context, before applying
compiler optimizations, to significantly speedup execution.

Proteus's cookie-cutter optimization through specialization is
**argument specialization**.
It folds arguments of functions to runtime constants for greater code
optimization during JIT compilation, by turbo-charging classical compiler
optimizations, such as loop unrolling, control flow simplification, and constant
propagation.

For GPU kernels, Proteus additionally sets the kernel execution
**launch bounds** dynamically, to optimize register allocation, while it also
folds to constants kernel launch dimensions (number of blocks and threads per
block).

To inteface with Proteus, users annotate functions (or GPU kernels) for JIT
compilation using the generic function attribute `annotate` with the "jit" tag,
and specify a list of function arguments to specialize for.

For example:
```cpp
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, int N, double *a, double *b)
{
  for(int i=0; i<N; ++i)
    a[i] = A*a[i] + b[i];
}
```
This attribute annotates the function `daxpy` for JIT specialization, folding
the function arguments 1 (A) and 2 (N) to runtime constants.

> [!NOTE]
> The argument list is **1-indexed**.

Proteus generates a unique specialization for each distinct set of runtime
values of the annotated function arguments and caches them in memory and on disk
to amortize JIT overhead within and across runs.

Proteus supports host, HIP, and CUDA compilation using Clang/LLVM (or compatible
vendor variants).
It is implemented as an LLVM plugin pass that extracts the bitcode of annotated
functions and instruments execution to pass runtime context information to its
runtime library for JIT compilation and runtime optimization.

## Building
The project uses `cmake` for building and depends on an LLVM installation
(CI tests cover LLVM 17, 18 and AMD ROCm versions 5.7.1, 6.2.1).
Check the top-level `CMakeLists.txt` for the available build options.
The typical building process is:
```
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install_path> ..
make install
```

The `scripts` directory contains setup scripts for different targets (host,
CUDA, ROCm) used on LLNL machines.
These scripts are good starting points to adapt for other machines too.
Run them from the repository root as follows:
```bash
source setup-<target>.sh
```
They load environment modules and create a `build-<target>` directory with a
working configuration for building.

> [!NOTE]
>️ On `lassen`, Proteus requires Clang CUDA compilation. We provide the script
> `scripts/build-llvm-cuda.sh` to build LLVM before executing
> `scripts/setup-cuda.sh`.

## Using

To integrate Proteus into your application, you must:

1. Annotate functions (or GPU kernels) for JIT specialization.
2. Modify your build to include the Proteus LLVM plugin pass and link with the runtime library.

This is done by adding Proteus's plugin pass to Clang compilation:
```bash
CXXFLAGS += -fpass-plugin=<install_path>/lib64/libProteusPass.so
```
and extending linker flags to include the runtime library (preferrably
rpath-ed) and LLVM libraries:
```bash
LDFLAGS += -L <install_path>/lib64 -Wl,-rpath,<install_path>/lib -lproteus $(llvm-config --libs)
```

A complete example is:
```bash
clang++ -fpass-plugin=<install_path>/lib64/libProteusPass.so \
-L <install_path>/lib64 -Wl,-rpath,<install_path>/lib64 -lproteus \
$(llvm-config --libs) MyAwesomeCode.cpp -o MyAwesomeExe
```

### CMake Integration

To use Proteus with CMake, make sure the Proteus install directory is in
`CMAKE_PREFIX_PATH`, or pass it as `-Dproteus_DIR=<install_path>`.
Then, in your project's `CMakeLists.txt` simply add the following two lines:

```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(target)
```

Where `target` is the name of your library or executable target.

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
- Tal Ben Nun, bennun2@llnl.gov
- Thomas Stitt, stitt4@llnl.gov

## License

Proteus is distributed under the terms of the Apache License (Version 2.0) with
LLVM Exceptions.

All new contributions must be made under the Apache-2.0 with LLVM Exceptions license.

See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

LLNL-CODE-2000857
