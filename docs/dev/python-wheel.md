# Building the Python Wheels

Proteus now builds Python wheels as a package family rather than a single
monolithic wheel:

- `proteus-python`
  - pure-Python shim package
  - installs the host backend by default
- `proteus-python-backend-host`
  - native host backend wheel for Linux and macOS
- `proteus-python-backend-cu12`
  - native CUDA 12 superset backend wheel for Linux `x86_64`

The user-facing import surface remains:

```python
import proteus
```

At runtime, the shim discovers installed backend entry points and prefers the
highest-priority backend:

1. `cuda12`
2. `host`

## Packaging Model

The shim package is built from the repository root with `setuptools`.

The backend packages are built from dedicated subprojects:

- `packaging/python/backend-host`
- `packaging/python/backend-cu12`

Those backend projects use `scikit-build-core` and the repository CMake build,
but they install their native payload into backend-local packages instead of
directly under `proteus/`:

- `proteus_backend_host/`
- `proteus_backend_cu12/`

Each backend package contains:

- backend-local `_proteus.*`
- backend-local `libproteus.*`
- repaired vendored LLVM/Clang shared libraries
- backend registration metadata under the `proteus.backends` entry-point group

The shim package exports:

- `proteus.active_backend`
- `proteus.available_backends()`
- the selected backend's native API re-exported from `proteus`

## Build-Time Requirements

All backend wheel builds require:

- Python with `build`
- CMake
- `pybind11`
- `scikit-build-core`
- a usable LLVM/Clang installation discovered via `LLVM_INSTALL_DIR`

Platform-specific requirements:

- macOS host backend:
  - Homebrew LLVM 22
  - `delocate`
- Linux host backend:
  - the `manylinux_2_28` LLVM container
  - `auditwheel`
- Linux CUDA backend:
  - the CUDA-capable `manylinux_2_28` LLVM container
  - CUDA Toolkit 12 with `libnvptxcompiler_static.a`
  - `auditwheel`

## Local Builds

### Shim Wheel

```bash
python3 -m venv /tmp/proteus-wheel-venv
/tmp/proteus-wheel-venv/bin/python -m pip install -U pip build setuptools setuptools-scm
/tmp/proteus-wheel-venv/bin/python -m build --wheel --outdir dist .
```

### Host Backend Wheel on macOS arm64

```bash
python3 -m venv /tmp/proteus-wheel-venv
/tmp/proteus-wheel-venv/bin/python -m pip install -U pip build scikit-build-core pybind11 delocate

brew install llvm@22

MACOSX_DEPLOYMENT_TARGET=14.0 \
LLVM_INSTALL_DIR=/opt/homebrew/opt/llvm \
  /tmp/proteus-wheel-venv/bin/python -m build --wheel \
  --outdir wheelhouse \
  packaging/python/backend-host
```

### Host Backend Wheel on Linux

```bash
bash packaging/python/image-scripts/build-manylinux-llvm-container.sh

python -m pip install -U pip build cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=ghcr.io/olympus-hpc/proteus-manylinux-llvm:22.1.3 \
  python -m cibuildwheel packaging/python/backend-host --output-dir wheelhouse
```

### CUDA 12 Backend Wheel on Linux

```bash
bash packaging/python/image-scripts/build-manylinux-cuda-llvm-container.sh

python -m pip install -U pip build cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=ghcr.io/olympus-hpc/proteus-manylinux-cuda-llvm:12.4.1-22.1.3 \
  python -m cibuildwheel packaging/python/backend-cu12 --output-dir wheelhouse
```

## Linux Container Images

The Linux wheel workflows use prebuilt GHCR images.

Host backend image inputs:

- `packaging/python/image-scripts/manylinux-llvm.Dockerfile`
- `packaging/python/image-scripts/build-llvm-manylinux.sh`
- `packaging/python/image-scripts/build-manylinux-llvm-container.sh`

CUDA backend image inputs:

- `packaging/python/image-scripts/manylinux-cuda-llvm.Dockerfile`
- `packaging/python/image-scripts/build-llvm-manylinux.sh`
- `packaging/python/image-scripts/build-manylinux-cuda-llvm-container.sh`

The CUDA image installs:

- LLVM `22.1.3`
- CUDA Toolkit `12.4`
- static `libnvptxcompiler_static.a`

## Installed-Wheel Verification

Do not treat build-tree imports as sufficient. Install from built wheels into a
clean environment.

### Default Host Install

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links dist \
  --find-links wheelhouse \
  dist/proteus_python-*.whl

export PROTEUS_CLANGXX_BIN=/path/to/clang++-22

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_std_headers.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
```

### CUDA Superset Install

Install the shim first, then the CUDA backend wheel from the local wheelhouse:

```bash
/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links dist \
  --find-links wheelhouse \
  dist/proteus_python-*.whl

/tmp/proteus-wheel-venv/bin/python -m pip install \
  --no-index \
  --find-links wheelhouse \
  proteus-python-backend-cu12==<version>
```

On CPU-only machines, validate the host path:

```bash
export PROTEUS_CLANGXX_BIN=/path/to/clang++-22

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_backend_loader.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_host_cpp_std_headers.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_invalid_clang_override.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_wheel_layout.py
```

On Linux systems with an NVIDIA GPU and driver, additionally validate:

```bash
export CUDA_HOME=/usr/local/cuda

/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_smoke.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_launch_validation.py
/tmp/proteus-wheel-venv/bin/python bindings/python/tests/test_gpu_cpp_pointer_validation.py
```

## Runtime Contract

### Host Backend

- ships Proteus and the LLVM/Clang runtime libraries it links against
- does not ship a host C++ compiler toolchain
- still requires a host `clang++` whose major version matches the bundled
  LLVM/Clang runtime for `frontend="cpp", target="host"`
- for the current wheel line, use `clang++-22` or equivalent and set
  `PROTEUS_CLANGXX_BIN` when it is not the default `clang++` on `PATH`
- a mismatched `clang++` may fail while resolving builtin/system headers during
  the in-process frontend compile

| Install | Backend | Target | Required compiler/toolchain |
| --- | --- | --- | --- |
| `pip install proteus-python` | `proteus-python[host]` | Host CPU | LLVM/Clang 22.x |
| `pip install proteus-python[cuda12]` | `proteus-python[cuda12]` | Host CPU + NVIDIA CUDA GPU | LLVM/Clang 22.x, or the NVIDIA toolchain when using NVCC for host/device compilation |

### CUDA 12 Backend

- is a superset backend: host functionality remains available
- does not vendor `libcuda.so`, `libcudart`, or the CUDA Toolkit
- requires an installed NVIDIA driver for CUDA functionality
- requires a matching CUDA 12 toolkit root for runtime compilation

CUDA toolkit resolution remains:

1. `PROTEUS_CUDA_HOME`
2. `CUDA_HOME`
3. `CUDA_PATH`

For wheel-targeted CUDA builds on Linux, Proteus now resolves `libcuda.so.1`
with `dlopen()` at runtime instead of linking the wheel directly against the
CUDA driver shared library. This keeps the wheel compatible with manylinux
repair and lets host-only functionality work on machines without an NVIDIA
driver.

## CI Workflow

The wheel workflow is defined in `.github/workflows/ci-wheels.yml`.

It produces three artifact groups:

- shim wheel
- host backend wheels
- CUDA backend wheels

Current target matrix:

- shim: pure Python
- host backend:
  - `macOS arm64`
  - `manylinux_2_28 x86_64`
- CUDA backend:
  - `manylinux_2_28 x86_64`

The workflow builds all backend wheels with `cibuildwheel`, then performs
explicit installed-wheel validation using the newly built shim and backend
wheels from the local artifact directories.
