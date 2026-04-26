# Building the Python Wheel

This page describes the first Python wheel workflow for Proteus.
It is intentionally scoped to the current minimal wheel target:

- CPU-only
- `libproteus` vendored into the wheel
- LLVM/Clang runtime libraries vendored during wheel repair
- LLVM 22 toolchains on both Linux and macOS
- host C++ JIT enabled, but requiring a host `clang++` at runtime
- no CUDA, HIP/ROCm, MLIR, or MPI in this wheel

The wheel path is designed for binary distribution first.
It does not try to replace the existing native CMake install flow for C++
consumers.

## Current Packaging Model

The Python package is built with `scikit-build-core` from
[`pyproject.toml`](../../pyproject.toml).
The CMake build is driven with these wheel-oriented settings:

- `BUILD_SHARED=ON`
- `PROTEUS_ENABLE_PYTHON=ON`
- `PROTEUS_PYTHON_WHEEL=ON`
- `PROTEUS_ENABLE_CUDA=OFF`
- `PROTEUS_ENABLE_HIP=OFF`
- `PROTEUS_ENABLE_MLIR=OFF`
- `PROTEUS_ENABLE_MPI=OFF`

The resulting package layout is:

- `proteus/__init__.py`
- `proteus/_proteus.*`
- `proteus/libproteus.*`
- repaired wheel payload for vendored LLVM/Clang libraries

The extension itself stays lightweight.
Shared library vendoring is handled by the platform repair step:

- `delocate` on macOS
- `auditwheel` on Linux

## Build-Time Requirements

Wheel builds still require an LLVM installation at build time.
Proteus discovers it through `LLVM_INSTALL_DIR`.

For wheel builds, this is a build-environment concern, not a user runtime
requirement.
The wheel is built against one pinned LLVM/Clang version per Proteus release.
The current wheel CI pin is LLVM `22.1.3` on Linux and Homebrew LLVM 22 on
macOS.

You currently need:

- Python with `build` and `scikit-build-core`
- CMake
- `pybind11`
- a usable LLVM/Clang installation
- platform repair tooling:
  - `delocate` on macOS
  - `auditwheel` on Linux

## Local Wheel Build

On macOS arm64, a typical local wheel build looks like this:

```bash
python3 -m venv /tmp/proteus-wheel-venv
/tmp/proteus-wheel-venv/bin/python -m pip install -U pip build scikit-build-core

brew install llvm

LLVM_INSTALL_DIR=/opt/homebrew/opt/llvm \
  /tmp/proteus-wheel-venv/bin/python -m build --wheel
```

This produces a wheel in `dist/`.

The raw wheel contains:

- the `proteus` Python package
- `_proteus` extension module
- `libproteus`

It does not yet contain vendored LLVM/Clang libraries until the repair step is
run.

## Repair Step

The repair step rewrites loader paths and copies non-system shared libraries
into the wheel.

### macOS

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install delocate
mkdir -p dist-repaired
/tmp/proteus-wheel-venv/bin/delocate-wheel \
  -w dist-repaired \
  -v dist/<wheel-name>.whl
```

After repair, the wheel includes vendored libraries under
`proteus/.dylibs/`, such as:

- `libLLVM.dylib`
- `libclang-cpp.dylib`
- any non-system secondary dependency such as `libzstd`

### Linux

The Linux wheel path is analogous, but uses `auditwheel`.
In CI, the Linux toolchain is built from source inside the
`manylinux_2_28` image by
[`packaging/wheels/build-llvm-manylinux.sh`](../../packaging/wheels/build-llvm-manylinux.sh),
which installs LLVM `22.1.3` into `/opt/llvm-22.1.3`.

```bash
bash packaging/wheels/build-llvm-manylinux.sh

LLVM_INSTALL_DIR=/opt/llvm-22.1.3 \
  python -m build --wheel
```

Repair is then handled with:

```bash
mkdir -p wheelhouse
auditwheel repair -w wheelhouse dist/<wheel-name>.whl
```

The repaired wheel should contain the vendored ELF dependencies that
`libproteus` and `_proteus` need at runtime.

## Installed-Wheel Verification

Do not treat build-tree import checks as sufficient.
The important validation step is to install the repaired wheel into a clean
environment and test that install directly.

A minimal local validation flow is:

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install --force-reinstall \
  dist-repaired/<wheel-name>.whl

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
```

These tests cover:

- import without `PYTHONPATH`
- host C++ JIT functionality
- explicit failure on bad `PROTEUS_CLANGXX_BIN`
- vendored library presence in the installed wheel

## Runtime Contract

This first wheel ships Proteus and the LLVM/Clang runtime libraries it links
against, but it does not ship a host C++ compiler toolchain.

For `frontend="cpp", target="host"`, Proteus resolves `clang++` at runtime in
this order:

1. `PROTEUS_CLANGXX_BIN`
2. LLVM-adjacent `clang++` hint
3. `PATH`

That means wheel users still need a compatible host `clang++` installed when
they use the host C++ JIT path.

## CI Workflow

The wheel workflow is defined in
[`ci-wheels.yml`](../../.github/workflows/ci-wheels.yml).

It currently targets:

- `macOS arm64`
- `manylinux_2_28 x86_64`

The workflow:

1. checks out the repository
2. provisions the pinned LLVM toolchain for the platform
3. builds LLVM from source on Linux inside the `manylinux_2_28` image
4. builds wheels with `cibuildwheel`
5. repairs wheels with `delocate` or `auditwheel`
6. uploads the wheel artifacts

On macOS, the workflow uses Homebrew `llvm` for LLVM 22.
On Linux, the workflow builds LLVM/Clang/MLIR from source for compatibility,
while the current wheel still keeps `PROTEUS_ENABLE_MLIR=OFF`.

The `cibuildwheel` test command runs the installed-wheel Python tests rather
than build-tree imports.

## Alternate Miniforge Workflow

There is also an alternate workflow at
[`ci-wheels-miniforge.yml`](../../.github/workflows/ci-wheels-miniforge.yml).

This workflow keeps the same wheel scope and macOS setup, but changes the Linux
toolchain provisioning strategy:

- Linux still builds in `manylinux_2_28`
- Miniforge supplies LLVM/Clang/MLIR, CMake, and Ninja
- the conda environment pins `sysroot_linux-64=2.28`
- `auditwheel show` is treated as the compatibility gate before repair

The Linux helper scripts for this path are:

- [`packaging/wheels/setup-miniforge-manylinux.sh`](../../packaging/wheels/setup-miniforge-manylinux.sh)
- [`packaging/wheels/auditwheel-repair-manylinux.sh`](../../packaging/wheels/auditwheel-repair-manylinux.sh)

The source-built LLVM workflow remains the baseline comparison path.

## Scope Boundaries

This wheel path is intentionally narrow.
It does not yet attempt to solve:

- CUDA wheel packaging
- HIP/ROCm wheel packaging
- MLIR-enabled wheels
- MPI-enabled wheels
- source-install UX for end users

Those are follow-on packaging tracks and should be treated separately from the
first CPU-only wheel.
