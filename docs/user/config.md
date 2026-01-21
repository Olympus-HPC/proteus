# Runtime Configuration

Proteus exposes a number of possible runtime configuration options, accessible
through environment variables.

| Environment Variable | Possible Values | Description |
| -------------------- | --------------- | ----------- |
| `PROTEUS_CACHE_DIR` | Directory path (default: `.proteus`) | Directory where cached JIT-compiled objects are stored |
| `PROTEUS_USE_STORED_CACHE` | `0` or `1` (default: `1`) | Enable the persistent storage cache |
| `PROTEUS_SPECIALIZE_LAUNCH_BOUNDS` | `0` or `1` (default: `1`) | Enable launch-bounds specialization on JIT kernels |
| `PROTEUS_SPECIALIZE_ARGS` | `0` or `1` (default: `1`) | Specialize JIT functions for input arguments |
| `PROTEUS_SPECIALIZE_DIMS` | `0` or `1` (default: `1`) | Specialize JIT kernels for launch dimensions |
| `PROTEUS_SPECIALIZE_DIMS_RANGE` | `0` or `1` (default: `0` on CUDA builds, `1` otherwise) | Specialize JIT kernels for launch-dimension ranges |
| `PROTEUS_CODEGEN` | `"rtc"`, `"serial"`, `"parallel"` (default: `"rtc"`) | Use RTC, serial, or parallel code generation |
| `PROTEUS_DISABLE` | `0` or `1` (default: `0`) | Disable Proteus (no JIT compilation) |
| `PROTEUS_DUMP_LLVM_IR` | `0` or `1` (default: `0`) | Dump LLVM IR of JIT modules in the directory `.proteus-dump` |
| `PROTEUS_RELINK_GLOBALS_BY_COPY` | `0` or `1` (default: `0`) | Relink device global variables at kernel launch instead of ELF patching |
| `PROTEUS_KERNEL_CLONE` | `"link-clone-prune"`, `"link-clone-light"`, `"cross-clone"` (default: `"cross-clone"`) | Cloning method for JIT module creation |
| `PROTEUS_ASYNC_COMPILATION` | `0` or `1` (default: `0`) | Enable asynchronous compilation |
| `PROTEUS_ASYNC_THREADS` | Integer `>= 1` (default: `1`) | Number of threads used for asynchronous compilation |
| `PROTEUS_AUTO_READONLY_CAPTURES` | `0` or `1` (default: `1`) | Enable automatic detection of read-only lambda captures for JIT specialization. When enabled, scalar captures (`int`, `float`, `double`, `bool`) that are read-only within the lambda body are automatically specialized without requiring explicit `jit_variable()` annotation. |
| `PROTEUS_ASYNC_TEST_BLOCKING` | `0` or `1` (default: `0`) | Make asynchronous compilation blocking for testing |
| `PROTEUS_ENABLE_TIMERS` | `0` or `1` (default: `0`) | Enable timer-based profiling output in JIT operations |
| `PROTEUS_TRACE_OUTPUT` | Semicolon-separated tokens: `specialization`, `ir-dump`, `kernel-trace`, `cache-stats` (default: empty) | Enable trace output. `specialization` prints specialization info, `ir-dump` dumps LLVM IR post-optimization, `kernel-trace` prints an end-of-run per-kernel summary with specialization and launch counts, and `cache-stats` prints object-cache hit/access statistics. Example: `"specialization;kernel-trace"` |
| `PROTEUS_ENABLE_TIME_TRACE` | `0` or `1` (default: `0`) | Enable time tracing for JIT operations |
| `PROTEUS_TIME_TRACE_FILE` | File path (default: empty) | Output file for time-trace JSON data |
| `PROTEUS_TIME_TRACE_GRAIN` | Integer microseconds `> 0` (default: `500`) | Minimum duration threshold passed to LLVM's time-trace profiler |
| `PROTEUS_DEBUG_OUTPUT` | `0` or `1` (default: `0`) | Log debug output information |
| `PROTEUS_OPT_PIPELINE` | String (default: unset) | String describing a middle-end `opt` pipeline, for example `default<O2>` |
| `PROTEUS_OPT_LEVEL` | `'0'`, `'1'`, `'2'`, `'3'`, `'s'`, `'z'` (default: `'3'`) | Default middle-end `opt` pipeline level; when unset, optimization defaults to `O3` |
| `PROTEUS_CODEGEN_OPT_LEVEL` | `'0'`, `'1'`, `'2'`, `'3'` (default: `'3'`) | Default back-end `llc` pipeline level; when unset, optimization defaults to `O3` |
| `PROTEUS_OBJECT_CACHE_CHAIN` | Comma-separated cache names: `"storage"`, `"mpi-local-lookup"`, `"mpi-remote-lookup"` (default: `"storage"`) | Configure the object cache chain. Valid caches: `storage` (persistent file cache), `mpi-local-lookup` (rank 0 writes, all ranks read from a shared filesystem, requires MPI build), `mpi-remote-lookup` (rank 0 writes and serves lookups over MPI, requires MPI build) |
| `PROTEUS_COMM_THREAD_POLL_MS` | Integer milliseconds (default: `25`) | Poll interval for the MPI remote-cache communication thread |
