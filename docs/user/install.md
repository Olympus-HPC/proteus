# Installation

Proteus is currently distributed through its Git repository, which you can use
to build and install the library.
We recommend using the latest `main` branch, which is well-tested, stable, and
includes the most recent features.

Building Proteus installs two components:
- the **Proteus LLVM plugin pass** (`ProteusPass`), and
- the **Proteus runtime library** (`libproteus`).

The LLVM plugin pass is necessary **only if** you use the Code annotation
interface and requires compiling your application with Clang, besides linking with `libproteus`.
If you the DSL or C++ frontend APIs then you only need to link your code with `libproteus`.
Both must be integrated into your application’s build system.
See [Integration](integration.md) section for more details on integrating with your build system.


## Building

Proteus uses `cmake` for building and requires an existing LLVM installation (CI
tests cover LLVM 18/19/20 with CUDA 12.2 and AMD ROCm versions 6.3.1, 6.4.1, 7.1.0).

The top-level `CMakeLists.txt` defines the following build options:

* `BUILD_SHARED`: build Proteus as a shared library (default is static).
* `ENABLE_TESTS`: build Proteus tests.
* `PROTEUS_ENABLE_HIP`: enable HIP support.
* `PROTEUS_ENABLE_CUDA`: enable CUDA support.

!!! info "Host, CUDA and HIP support"
    Proteus always supports host JIT compilation.
    You can additionally enable CUDA or HIP JIT compilation by setting
    `PROTEUS_ENABLE_CUDA` or `PROTEUS_ENABLE_HIP` respectively.

A typical build looks like this:
```shell
git clone https://github.com/Olympus-HPC/proteus.git
cd proteus
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install-path> <other options> ..
make -j install
```

## Testing

We recommend enabling tests when deploying Proteus on a new machine and running them once to verify the installation:
```shell
cd build
make test
```

If you encounter bugs or issues, please let us know via the
[Github issue tracker](https://github.com/Olympus-HPC/proteus/issues).
