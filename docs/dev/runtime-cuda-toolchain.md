# Runtime CUDA Toolchain Detection

## Overview

Proteus no longer bakes absolute CUDA toolkit paths such as `libdevice` or the
CUDA runtime library directory into `libproteus` at CMake configure time.

Instead, CUDA builds resolve the runtime toolkit from the process environment on
first use in
[`src/runtime/Frontend/CUDAToolchain.cpp`](../../src/runtime/Frontend/CUDAToolchain.cpp).

This affects two CUDA compilation paths:

1. Device-side compilation paths that need `libdevice`.
2. Host+CUDA shared-library JIT paths that need `libcudart` registration
   symbols such as `__cudaRegisterFatBinary`.

## Resolution Order

### CUDA toolkit root

Proteus looks for a CUDA root in this order:

1. `PROTEUS_CUDA_HOME`
2. `CUDA_HOME`
3. `CUDA_PATH`

If none are set, runtime CUDA compilation fails.

### Explicit `libdevice`

`PROTEUS_CUDA_LIBDEVICE_PATH` can override the `libdevice` bitcode file
directly.

If a CUDA root is also available, Proteus still resolves the runtime CUDA
library directory from that root for Host+CUDA shared-library compilation.

## Resolved Artifacts

Once a root is found, Proteus derives:

1. `libdevice`
   `nvvm/libdevice/libdevice.10.bc`
2. CUDA runtime library directory
   first matching directory under the root from:
   - `lib64`
   - `lib`

The runtime library directory is used only for Host+CUDA shared-library
compilation, where Clang still needs explicit `-L... -lcudart` linkage.

## Version Check

CUDA builds embed the build-time `CUDAToolkit_VERSION` as a compile definition.

At runtime, Proteus reads the selected toolkit version from:

1. `version.json`
2. `version.txt`

Proteus compares only the major version between:

1. the toolkit used to build Proteus
2. the toolkit root selected at runtime

If the major versions differ, runtime CUDA toolchain resolution fails.

This is a coarse guard against mixing toolchains from different CUDA release
families while still tolerating minor and patch differences.

## Why This Exists

The previous wheel-oriented behavior embedded builder-local CUDA filesystem
paths into `libproteus`. That is fragile for:

1. Python wheels moved to another machine
2. nonstandard CUDA install prefixes
3. cluster environments where CUDA is injected by environment modules

Runtime resolution makes the wheel and the shared library relocatable, provided
the process environment points at a compatible CUDA toolkit.

## Cache Interaction

CUDA toolchain resolution happens only when Proteus must compile new code.

If a test or program hits the object cache, it may succeed without touching the
runtime CUDA resolver at all. This can mask missing `CUDA_HOME`-style
environment variables.

In practice:

1. warm cache: may succeed without a CUDA root env var
2. cold cache: will fail if no CUDA root env var is set

When debugging, clear the cache first:

```bash
rm -rf .proteus
```

Or set an explicit cache location:

```bash
export PROTEUS_CACHE_DIR=/tmp/proteus-cache
```

## Typical Cluster Usage

On systems that provide CUDA through environment modules, loading the module is
usually enough because it exports `CUDA_HOME`:

```bash
ml load cuda/12
```

After that, both Python and standalone frontend tests can resolve the runtime
CUDA toolkit without relying on build-time absolute paths.
