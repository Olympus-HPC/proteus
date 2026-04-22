# Optimization Pipeline

This page documents where `PROTEUS_OPT_PIPELINE` applies inside Proteus.

`PROTEUS_OPT_PIPELINE` is a textual LLVM pass pipeline. It is used when Proteus
owns LLVM IR optimization, or when Proteus can forward the textual pipeline to
an LLVM API such as LTO. It is not a generic compiler flag for external
compiler paths such as NVCC, HIPRTC, or Clang shared-library compilation.

## Selection Order

Proteus selects the optimization pipeline from the effective
`CodeGenerationConfig`:

1. A per-kernel JSON `"Pipeline"` from `PROTEUS_TUNED_KERNELS`, when the code
   path looks up config by kernel name.
2. The global `PROTEUS_OPT_PIPELINE` environment variable.
3. The default pipeline implied by `PROTEUS_OPT_LEVEL`.

Per-kernel JSON `"Pipeline"` currently applies to annotated CUDA/HIP runtime
JIT paths because they call:

```cpp
Config::get().getCGConfig(KernelName)
```

Frontend module compilation currently uses the global config:

```cpp
Config::get().getCGConfig()
```

so per-kernel JSON `"Pipeline"` does not naturally apply to DSL, MLIR, or C++
frontend modules.

## Support Matrix

| Frontend / API path | Target type | Main compile path | Uses `PROTEUS_OPT_PIPELINE`? | Notes |
| --- | --- | --- | --- | --- |
| Annotated C/C++ JIT | Host CPU | `JitEngineHost::compileAndLink()` -> `compileOnly()` | Yes | Uses global `CodeGenerationConfig`. Cache key includes codegen config and runtime specialization config. |
| Annotated CUDA kernel JIT | CUDA device | `JitEngineDeviceCUDA` / `CompilationTask` | Yes | Uses named per-kernel `CodeGenerationConfig`. Explicit LLVM IR optimization before device codegen. |
| Annotated HIP kernel JIT | HIP, `PROTEUS_CODEGEN=serial` | `CompilationTask` -> `optimizeIR()` -> serial codegen | Yes | Uses named per-kernel `CodeGenerationConfig`. Explicit LLVM IR optimization before serial device codegen. |
| Annotated HIP kernel JIT | HIP, `PROTEUS_CODEGEN=parallel` | `CompilationTask` -> HIP LTO codegen | Yes | Custom pipeline is forwarded to `llvm::lto::Config::OptPipeline`. |
| Annotated HIP kernel JIT | HIP, `PROTEUS_CODEGEN=rtc` | `CompilationTask` -> HIPRTC link/compile | No | HIPRTC accepts some compiler options, but does not expose a documented textual LLVM pass pipeline equivalent. |
| DSL `JitModule`, LLVM backend | Host CPU | DSL -> LLVM IR -> host dispatcher | Yes | Host dispatcher reaches `JitEngineHost::compileOnly()`. Module hash includes codegen config. |
| DSL `JitModule`, LLVM backend | CUDA device | DSL -> LLVM IR -> CUDA dispatcher | Yes | CUDA dispatcher reaches `JitEngineDeviceCUDA::compileOnly()`. Module hash includes codegen config. |
| DSL `JitModule`, LLVM backend | HIP device | DSL -> LLVM IR -> HIP dispatcher | Yes | HIP dispatcher reaches `JitEngineDeviceHIP::compileOnly()`. Module hash includes codegen config. |
| DSL `JitModule`, MLIR backend | Host CPU | MLIR -> LLVM IR -> host dispatcher | Yes | Same dispatcher path as LLVM backend. Module hash includes codegen config. |
| DSL `JitModule`, MLIR backend | CUDA device | MLIR -> LLVM IR -> CUDA dispatcher | Yes | Same CUDA dispatcher path. Module hash includes codegen config. |
| DSL `JitModule`, MLIR backend | HIP device | MLIR -> LLVM IR -> HIP dispatcher | Yes | Same HIP dispatcher path. Module hash includes codegen config. |
| Direct `MLIRJitModule` | Host CPU | MLIR source -> LLVM IR -> host dispatcher | Yes | Uses dispatcher compile path. Module hash includes codegen config. |
| Direct `MLIRJitModule` | CUDA device | MLIR source -> LLVM IR -> CUDA dispatcher | Yes | Uses dispatcher compile path. Module hash includes codegen config. |
| Direct `MLIRJitModule` | HIP device | MLIR source -> LLVM IR -> HIP dispatcher | Yes | Uses dispatcher compile path. Module hash includes codegen config. |
| `CppJitModule`, Clang backend | Host CPU | Clang emits LLVM IR -> host dispatcher | Yes | Clang emits optimized-mode IR with `-O3 -Xclang -disable-llvm-passes`; Proteus runs the configured pipeline. |
| `CppJitModule`, Clang backend | CUDA device-only | Clang emits device LLVM IR -> CUDA dispatcher | Yes | Proteus optimizes the emitted LLVM IR. Module hash includes codegen config. |
| `CppJitModule`, Clang backend | HIP device-only | Clang emits device LLVM IR -> HIP dispatcher | Yes | Proteus optimizes the emitted LLVM IR. Module hash includes codegen config. |
| `CppJitModule`, Clang backend | Host+CUDA | Clang compiles mixed offload translation unit directly to shared library | No | Proteus receives a final `.so`, not host/device LLVM IR. |
| `CppJitModule`, Clang backend | Host+HIP | Clang compiles mixed offload translation unit directly to shared library | No | Same reason as Host+CUDA. |
| `CppJitModule`, NVCC backend | CUDA device-only | NVCC emits cubin | No | NVCC owns optimization and codegen. Cache key intentionally does not include Proteus codegen config. |
| `CppJitModule`, NVCC backend | Host+CUDA | NVCC emits shared library | No | Proteus receives a final binary artifact. Cache key intentionally does not include Proteus codegen config. |

## Cache Key Policy

Cache keys include the configuration that can affect the generated artifact.

Frontend module caches use a codegen-only hash:

```cpp
hashCodeGenConfig(CGConfig)
```

This includes:

- `PROTEUS_CODEGEN`
- `PROTEUS_OPT_LEVEL`
- `PROTEUS_CODEGEN_OPT_LEVEL`
- `PROTEUS_OPT_PIPELINE`

Runtime annotated JIT cache keys also include runtime specialization policy:

```cpp
hashRuntimeSpecializationConfig(CGConfig)
```

This includes:

- `PROTEUS_SPECIALIZE_ARGS`
- `PROTEUS_SPECIALIZE_DIMS`
- `PROTEUS_SPECIALIZE_DIMS_RANGE`
- `PROTEUS_SPECIALIZE_LAUNCH_BOUNDS`

Those specialization flags are not part of generic frontend module cache keys
because frontend modules do not use runtime specialization policy in the same
way annotated runtime JIT paths do.

## Known Exclusions

### HIP RTC

`PROTEUS_CODEGEN=rtc` routes HIP compilation through HIPRTC. The HIPRTC linker
interface accepts some compiler options, but it does not expose a documented
LLVM textual pass pipeline interface equivalent to `opt`/PassBuilder or LLVM
LTO's `OptPipeline`.

### CppJit Host+CUDA / Host+HIP

The Clang backend currently compiles mixed host/device C++ offload source
directly into a shared library for `HOST_CUDA` and `HOST_HIP`. Proteus receives
the final `.so`, so it cannot run its LLVM pass pipeline over the host and
device modules.

### NVCC Backend

The NVCC backend is an external compiler path. Proteus does not own LLVM IR
optimization there, so `PROTEUS_OPT_PIPELINE` does not apply and is not included
in NVCC CppJit cache keys.
