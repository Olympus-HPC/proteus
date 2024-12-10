![License: Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)

# Proteus

Proteus optimizes C/C++ code execution, including CUDA/HIP kernels, by applying
runtime optimizations using Just-In-Time (JIT) compilation powered by LLVM.

## Description
The motivation is that JIT compilation exposes runtime information that is
unavailable during ahead-of-time (AOT) compilation, for example, the exact
runtime values of variables during program execution.
Proteus uses runtime information to *specialize* generated code using JIT
compilation can significantly speedup execution.

Proteus supports optimizing the execution of functions (or kernels for GPUs)
using runtime specialization.
Our cookie-cutter optimization through specialization at the moment is
**argument specialization**.
Proteus folds arguments of functions to runtime constants, which enables greater
code optimization during JIT code generation by turbo-charging classical
compiler optimizations, such as loop unrolling, control flow simplification, and
constant propagation.
For GPU kernels, Proteus additionally sets the kernel execution **launch
bounds** dynamically, to optimize register allocation and GPU concurrency.

To inteface with Proteus, users annotate functions (or GPU kernels) for JIT
compilation using the generic attribute `annotate` with the "jit" tag, to
specify a list of function arguments to specialize for.

For example:
```cpp
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, int N, double *a, double *b)
{
  for(int i=0; i<N; ++i)
    a[i] = A*a[i] + b[i];
}
```
The attribute annotates the function `daxpy` for JIT specialization folding the
function arguments 1 (A) and 2 (N) to runtime constants.

👉 Note the argument list is **1-indexed**.

Proteus will create a new, unique specialization at runtime for each set of
unique runtime values of the annotated function arguments.
This means that the same function can have multiple, unique specializations
corresponding to different runtime values.
Proteus implements in-memory caching so it will compile once and re-use
specializations to amortize the JIT compilation overhead.
Also, Proteus implements a persistent cache which stores specializations on disk
to make them re-usable across program runs.

Proteus supports host, HIP, and CUDA compilation using Clang/LLVM or
compatible vendor variants.
Proteus extracts the bitcode of annotated functions and instruments execution to
read runtime values of arguments using a plugin LLVM pass.
Runtime JIT compilation is implemented by the Proteus runtime library based on
LLVM.

## Building
The project uses `cmake` for building and depends on an LLVM installation
(upstream versions 16, 17 or AMD ROCm version 5.7.1 have been tested).
Check the top-level `CMakeLists.txt` for the available build options.
The typical building process is:
```
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install_path> ..
make install
```

The directory `scripts` contains scripts to setup building on the `tioga` and
`lassen` machines at LLNL.
Run them at the top-level directory as `source setup-<machine>.sh` to load
environment modules and create a `build-<machine>` directory with a working
configuration for the machine.

⚠️ For `lassen`, Proteus needs Clang CUDA compilation so we provide a script to
build LLVM in `scripts/build-llvm-lassen.sh` to manually run before
`scripts/setup-lassen.sh`.

## Using

Besides annotating the source code, users **must** modify the compilation of
their application to include the Proteus JIT plugin pass in the compilation
options and link with the JIT runtime library.
This is done by adding `-fpass-plugin=<install_path>/lib/libProteusJitPass.so`
to Clang compilation and extend linker flags to include the runtime library (preferrably
rpath-ed) as in `-L <install_path>/lib -Wl,-rpath,<install_path>/lib
-lproteusjit`.

An one-liner example is:
```
clang++ -fpass-plugin=<install_path>/lib/libiProteusJitPass.so -L <install_path>/lib -Wl,-rpath,<install_path>/lib -lproteusjit MyAwesomeCode.cpp -o MyAwesomeExe
```

### CMake

To use Proteus with CMake, make sure the Proteus install directory is defined
in the `CMAKE_PREFIX_PATH` environment variable. Then, in your project's 
`CMakeLists.txt` simply add the following two lines:

```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(target)
```

Where `target` is the name of your library or executable target.


## Contributing

We welcome contributions to Proteus in the form of pull requests targeting the
`main` branch of the repo, as well as questions, feature requrests, or bug reports
via issues.

## Code of Conduct

Please note that Proteus has a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in the Proteus
community, you agree to abide by its rules.

## Authors
Proteus was created by Giorgis Georgakoudis, georgakoudis1@llnl.gov.

Key contributors in code or design are David Beckingsale, beckingsale1@llnl.gov
and Konstantinos Parasyris, parasyris1@llnl.gov.

## License

Proteus is distributed under the terms of the Apache License (Version 2.0) with
LLVM Exceptions.

All new contributions must be made under the Apache-2.0 with LLVM Exceptions license.

See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

LLNL-CODE-2000857
