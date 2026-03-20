# Concepts

This page is a developer-facing architecture overview of Proteus.
The [user guide](../user/interface.md) explains how to use the public
interfaces; this page explains how those interfaces map to the implementation.

## What Proteus Is

Proteus is a runtime specialization and JIT layer for host, CUDA, and HIP code.
Its core idea is simple: capture runtime constants, specialize code with those
values, then run the usual LLVM-style optimization and code-generation flow on
that specialized program.

At a high level, Proteus sits between an application and the final compiled
artifact that executes.
Instead of treating runtime values as opaque inputs, Proteus can fold selected
values into the program being compiled, which often enables stronger
optimizations than ahead-of-time compilation alone.

## Two Core Build Products

Proteus is built around two main products:

- `ProteusPass`: the LLVM pass used by the annotation interface
- `libproteus`: the runtime library that performs JIT compilation, caching,
  loading, and execution

The split matters because the three user-facing interfaces do not all enter the
system the same way.
The annotation interface uses both the pass and the runtime library: the pass
finds annotated regions during normal compilation and rewrites them to route
through the runtime.
The DSL and C++ frontend APIs skip the pass and build JIT units directly through
runtime library APIs.

Conceptually, the pass handles compile-time instrumentation, while the runtime
library handles runtime compilation and execution.

## Three Frontends, One Runtime Pipeline

Proteus exposes three main ways to define JIT work:

- **Code annotations**: source code is compiled ahead of time, and the pass
  extracts and instruments JIT-annotated regions
- **DSL API**: code is constructed programmatically as IR through the frontend
  builders
- **C++ frontend API**: source strings are compiled on demand through Clang or,
  for CUDA, optionally NVCC

These paths look different at the API level, but they converge on the same core
runtime machinery.
Each path ultimately produces a JIT unit that is specialized, hashed, looked up
in caches, compiled if necessary, and then executed through a target-specific
dispatcher.

For interface-level details, see the [user interface guide](../user/interface.md).

## Compiler Stack Integration

Proteus is not a replacement compiler stack.
It is a specialization and orchestration layer built on top of existing compiler
components, and different parts of Proteus rely on different layers of that
stack.

- **LLVM**: the main substrate for Proteus. The pass operates on LLVM IR, and
  the runtime ultimately specializes, optimizes, and compiles LLVM-based IR for
  execution.
- **MLIR**: an optional frontend path for the DSL. When enabled, Proteus can
  build code in MLIR first and then lower it into LLVM IR before continuing
  through the normal runtime pipeline.
- **Clang**: used both ahead of time and at runtime. The annotation interface
  depends on Clang-compatible compilation so the pass can observe annotations,
  and the C++ frontend can invoke Clang to compile source strings on demand.
- **NVCC**: an optional CUDA-only backend for the C++ frontend. In CUDA builds,
  Proteus can compile source strings through NVCC instead of Clang when that
  toolchain is a better fit.
- **HIP / hiprtc**: the ROCm-side compiler/runtime components used for
  HIP-enabled flows, especially when Proteus needs to materialize or launch GPU
  work for AMD targets.
- **LLD**: part of the HIP-enabled toolchain integration, where Proteus relies
  on linker support when assembling and loading device-oriented artifacts.

A useful rule of thumb is that Proteus owns specialization, caching, dispatch,
and JIT orchestration, while external compiler components provide the parsing,
IR infrastructure, lowering, compilation, and linking machinery underneath.

## Execution Model

Across the three frontends, the common lifecycle is:

1. identify or build a JIT unit
2. collect runtime constants or other specialization inputs
3. derive a content-based module hash
4. check the configured caches for an existing compiled artifact
5. compile when there is a cache miss
6. load and run the result through the dispatcher for the selected target

The main difference between the frontends is how the JIT unit is created.
The annotation path starts from ahead-of-time compiled code and produces
instrumented stubs that call into the runtime.
The DSL and C++ frontend paths construct modules directly in the runtime,
either as IR or as source-to-compile requests.

Host and device execution follow the same high-level model.
What changes is the backend work required to materialize and launch the compiled
artifact.
On the host, Proteus resolves and calls compiled functions directly.
On CUDA and HIP, Proteus also has to manage device-side loading and kernel
launch through the appropriate runtime path.

## Target Models and Dispatch

Proteus supports three target families:

- host
- CUDA
- HIP

Dispatch is target-specific.
Once a JIT unit has been specialized and compiled, the runtime selects the
appropriate dispatcher implementation for the target and uses it to load,
resolve, and execute the compiled artifact.

The dispatch layer is what lets the higher-level interfaces share one runtime
pipeline while still supporting very different execution environments.
The frontends do not need to reimplement launch logic for each target; instead,
they hand execution off to the dispatcher layer.

Some GPU-facing frontends do not launch device code by calling a raw kernel
entry point directly. Instead, Proteus can first build a small host-side
launcher that accepts normal launch parameters such as grid size, block size,
shared-memory size, and stream, and then forwards the launch through the
CUDA or HIP runtime. This keeps the frontend API uniform while still letting
the dispatcher handle target-specific kernel loading and launch details.

## Caching and Reuse

Proteus uses caching at multiple levels.
In-memory reuse avoids repeating work within a process, while persistent object
caching allows compiled artifacts to survive across runs.

Compiled artifacts are keyed by a content-derived hash.
That hash reflects the specialized program being compiled, so equivalent work
can be reused without recompilation.
At runtime, the dispatcher consults a configurable cache chain before invoking
compilation.

In MPI-enabled builds, the cache chain can also include distributed cache
behaviors in addition to local persistent storage.
Those layers are still part of the same overall idea: compiled artifacts are
looked up first and only rebuilt when no cache level can serve them.

For runtime knobs that control cache behavior, see the
[runtime configuration guide](../user/config.md).

## Codebase Map for Contributors

When navigating the implementation, these areas are the most useful starting
points:

- `src/pass`: annotation discovery, host/device instrumentation, and pass-side
  extraction logic
- `src/runtime`: JIT engines, dispatch, caching, and runtime services
- `src/runtime/Frontend`: the DSL and C++ frontend implementations that build
  JIT work directly through the runtime
- `include/proteus`: the public API surface exposed to applications

A useful mental model is:

- `src/pass` explains how annotated code enters the system
- `src/runtime` explains how specialized code is compiled, cached, and executed
- `include/proteus` explains what the public interfaces promise to callers

## Suggested Reading Path

For a new contributor, a good reading order is:

1. this page for the architectural overview
2. [user interface guide](../user/interface.md) for the surface APIs
3. [runtime configuration guide](../user/config.md) for runtime controls and
   cache-related settings
4. [API reference](api.md) for symbol-level detail
