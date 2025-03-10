# Installation

Currently, Proteus is distributed using its git repo to build and install it.
We recommend using the **latest main** branch version, which is well tested and
robust while including the most recent features.

Proteus builds and installs two components: the Proteus LLVM plugin pass
(ProteusPass) and the Proteus runtime library (libproteus).
The user must integrate both of them in their application build system.
We provide information on how to integrate Proteus with your application in the
[Integration](integration.md) section.

## Building

The project uses `cmake` for building and depends on an LLVM installation (CI
tests cover LLVM 17, 18 and AMD ROCm versions 5.7.1, 6.2.1).  The top-level
`CMakeLists.txt` has the following (binary) build options:

* `BUILD_SHARED`: builds Proteus as a shared library (default is static).
* `ENABLE_TESTS`: builds Proteus tests.
* `PROTEUS_ENABLE_HIP`: enables HIP support.
* `PROTEUS_ENABLE_CUDA`: enable CUDA support.
* `PROTEUS_ENABLE_DEBUG`: logs debugging information (for developers).
* `PROTEUS_ENABLE_TIME_TRACING`: stores a time trace file in JSON format for Proteus operations using flame graphs.

!!! info "Host, CUDA and HIP support"
    Proteus supports host JIT compilation in all cases.
    On top of that, it supports either CUDA or HIP JIT compilation, setting 
    `PROTEUS_ENABLE_CUDA` or `PROTEUS_ENABLE_HIP` respectively.

A typical build process is:
```shell
git clone https://github.com/Olympus-HPC/proteus.git
cd proteus
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install-path> <other options> ..
make -j install
```

## Testing

It is advised to enable tests when deploying Proteus on a machine for the first
time and run them:
```shell
cd build
make test
```

We are keen to help with bugs or any other issues in our repo's
[issues](https://github.com/Olympus-HPC/proteus/issues) page!