# Runtime Configuration

Proteus exposes a number of possible runtime configuration options, accessible
through environment variables.

| Environment Variable             | Possible Values     | Description                                                               |
| -------------------------------- | ------------------- | ------------------------------------------------------------------------- |
| `PROTEUS_USE_STORED_CACHE`       | 0 or 1 (default: 1) | Enable the persistent storage cache                                       |
| `PROTEUS_SET_LAUNCH_BOUNDS`      | 0 or 1 (default: 1) | Set or not launch bouds on JIT kernels                                    |
| `PROTEUS_SPECIALIZE_ARGS`        | 0 or 1 (default: 1) | Specialize JIT functions for input arguments                              |
| `PROTEUS_SPECIALIZE_DIMS`        | 0 or 1 (default: 1) | Specialize JIT kernels for launch dimensions                              |
| `PROTEUS_USE_HIP_RTC_CODEGEN`    | 0 or 1 (default: 1) | Use HIP RTC for machine code generation (instead of our own) 
| `PROTEUS_DISABLE`                | 0 or 1 (default: 0) | Disable Proteus (no JIT compilation)                                      |
| `PROTEUS_DUMP_LLVM_IR`           | 0 or 1 (default: 0) | Dump the LLVM IR of JIT modules in the directory `.proteus-dump`          |
| `PROTEUS_RELINK_GLOBALS_BY_COPY` | 0 or 1 (default: 0) | Relink device global variables at kernel launch (instead of ELF patching) |
| `PROTEUS_ASYNC_COMPILATION`      | 0 or 1 (default: 0) | Enable asynchronous compilation                                           |
| `PROTEUS_ASYNC_THREADS`          | >=1 (default: 1)    | Set number of threads for asynchronous compilation                        |
| `PROTEUS_ASYNC_TEST_BLOCKING`    | 0 or 1 (default: 0) | Make asynchronous compilation blocking for testing                        |
