# C/C++ JIT using LLVM

This repository implements Just-In-Time (JIT) compilation in C/C++ for runtime optimization.

## Description
The motivation is to use JIT compilation for optimizing the execution of functions (or kernels for GPUs) using runtime specialization.
Our cookie-cutter optimization through specialization at the moment is **runtime constant propagation**.
Folding runtime values to constants enables multiple helpful effects for code optimization
(loop unrolling, code elimination).
Users annotate functions for JIT compilation using the generic attribute `annotate` and specify a list of function
arguments that are treated as runtime constants.
The JIT compiler folds runtime arguments to constants and compiles an optimized specialization of the function.

For example:
```cpp
__atribute__((annotate("jit", 1, 2)))
void daxpy(double A, int N, double *a, double *b)
{
  for(int i=0; i<N; ++i)
    a[i] = A*a[i] + b[i];
}
```
The attribute annotates the function `daxpy` for JIT specialization folding the function arguments 1 (A) and 2 (N) to runtime
constants.

👉 Note the argument list is **1-indexed**.

For each tuple of unique runtime values of the annotated function arguments the JIT compiler will create a new 
unique specialization. The same function can have multiple, unique specializations corresponding to different runtime values.
The JIT compiler implements in-memory caching so it will compile once and re-use specializations to amortize the JIT compilation overhead.
Currently, JIT supports host, HIP, and CUDA compilation using Clang/LLVM or compatible vendor variants.
JIT compilation is implemented as a plugin LLVM pass supported by a JIT runtime library using LLVM.

## Building
The project uses `cmake` for the building system and depends on an LLVM installation
(upstream versions 16, 17 or AMD ROCm version 5.7.1 have been tested).
Check the top-level `CMakeLists.txt` for the available build options.
The typical building process is:
```
mkdir -p build && cd build
cmake ..
make install
```

The directory `scripts` contains scripts to setup building on the `tioga` and `lassen` machines at LLNL.
Run them at the top-level directory as `source setup-<machine>.sh` to load environment modules
and create a `build-<machine>` directory with a working configuration for the machine.

⚠️ For `lassen`, JIT needs Clang CUDA compilation so we provide a script to build LLVM
in `scripts/build-llvm-lassen.sh` to manually run before `scripts/setup-lassen.sh`.

## Using

Besides annotating the source code, users **must** modify their compilation to include the JIT plugin pass
and link with the JIT runtime library.
This is done by adding `-fpass-plugin=<install_path>/lib/libJitPass.so` to the Clang compilation arguments and
pointing to the library (preferrably rpath-ed) as in `-L <install_path>/lib -Wl,-rpath,<install_path>/lib -ljit`.
An one-liner example is:
```
clang++ -fpass-plugin=<install_path>/lib/libJitPass.so -L <install_path>/lib -Wl,-rpath,<install_path>/lib -ljit MyAwesomeCode.cpp -o MyAwesomeExe 
```

🚧 Create a cmake file for integrating to projects
