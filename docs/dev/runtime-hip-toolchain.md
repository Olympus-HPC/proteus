# Runtime HIP Asset Detection

## Overview

Proteus no longer bakes the ROCm device-library directory into `libproteus` at
CMake configure time.

Instead, HIP builds resolve the ROCm installation from the process environment
on first use in `src/runtime/Frontend/HIPToolchain.cpp`.

This currently affects the HIP device compilation path in
`src/include/proteus/impl/Frontend/DispatcherHIP.h`, where Proteus links ROCm
bitcode such as `ocml.bc`, `ockl.bc`, and the selected `oclc_*` configuration
modules into the generated LLVM IR before HIPRTC code generation.

## Resolution Order

Proteus looks for a ROCm root in this order:

1. `PROTEUS_ROCM_PATH`
2. `ROCM_PATH`

If neither is set, runtime HIP compilation fails.

## Resolved Assets

Once a root is found, Proteus derives the ROCm device-library directory from:

1. `amdgcn/bitcode`

under the selected ROCm root.

The existing runtime selection logic for the bitcode files themselves is
unchanged:

1. `ocml.bc`
2. `ockl.bc`
3. the newest available `oclc_abi_version_*.bc`
4. the device-specific `oclc_isa_version_<arch>.bc`
5. the default math and wavefront policy bitcode modules

## Version Check

HIP builds embed the build-time `hip_VERSION` as a compile definition.

At runtime, Proteus reads the selected installation version from:

1. `.info/version`
2. `include/hip/hip_version.h`
3. `include/hip/amd_detail/amd_hip_version.h`

Proteus compares the major and minor versions between:

1. the HIP installation used to build Proteus
2. the ROCm root selected at runtime

If either component differs, runtime HIP asset resolution fails.

## Why This Exists

The previous behavior embedded a builder-local ROCm bitcode path into
`libproteus`. That is fragile for:

1. installed shared libraries moved to another machine
2. cluster environments where ROCm is injected by environment modules
3. systems where the runtime ROCm prefix differs from the build machine

Runtime resolution makes HIP asset lookup relocatable, provided the process
environment points at a compatible ROCm installation.

## Cache Interaction

HIP toolchain resolution happens only when Proteus must compile new HIP code.

If a program hits the object cache, it may succeed without touching the runtime
resolver at all. This can mask missing `ROCM_PATH`-style environment
variables.

When debugging, clear the cache first:

```bash
rm -rf .proteus
```

Or set an explicit cache location:

```bash
export PROTEUS_CACHE_DIR=/tmp/proteus-cache
```

## Typical Cluster Usage

On systems that provide ROCm through environment modules, loading the module is
usually enough because it exports `ROCM_PATH`:

```bash
ml load rocm/7.1.1
```

After that, HIP frontend tests and applications can resolve the runtime ROCm
device libraries without relying on build-time absolute paths.
