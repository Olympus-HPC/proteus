# Proteus Component Architecture

```
                                   +-----------------+
                                   |                 |
                                   |   CodeContext   |
                                   |                 |
                                   +-----------------+
                                            ^
                                            |
                     +-----------------+    |    +-----------------+
                     |                 |    |    |                 |
                     |      Code       |<---+--->|      Cache      |
                     |                 |         |                 |
                     +-----------------+         +-----------------+
                              ^                           ^
                              |                           |
                     +--------+--------+         +--------+--------+
                     |                 |         |                 |
                     |      Engine     |<------->| CompilationTask |
                     |                 |         |                 |
                     +-----------------+         +-----------------+
                        |           ^                    ^
                        |           |                    |
                        v           |                    |
                     +-------------------+               |
                     |                   |               |
                     |      Builder      |---------------+
                     |                   |
                     +-------------------+
                              |
                              v
                     +-----------------+
                     |                 |
                     |CompilationResult|
                     |                 |
                     +-----------------+
```

## Component Responsibilities

1. **Engine**: Encapsulates backend details and creates compilation tasks
   - Creates CompilationTasks with appropriate specialization parameters
   - Coordinates between Code, Cache, and Builder components
   - Backend-specific implementations (CPU, CUDA, HIP)

2. **CompilationTask**: Captures all compilation parameters
   - Contains function code, runtime constants, dimensions, and configuration
   - Represents a specific compilation request with unique hash

3. **Builder**: Transforms CompilationTask into CompilationResult
   - Handles IR specialization, optimization, and code generation
   - SyncBuilder (for synchronous compilation)
   - AsyncBuilder (for thread-pool based asynchronous compilation)

4. **CompilationResult**: Contains compiled function and metadata
   - Function pointer for execution
   - Metadata about the compilation
   - Optional object code buffer

5. **Cache**: Stores and retrieves compilation results
   - Memory-based, disk-based, or hierarchical implementation
   - Indexed by hash values for fast lookup

6. **Code**: Represents the function code to be specialized
   - Contains LLVM IR module with function to be compiled
   - Provides hash for identifying code

7. **CodeContext**: Tracks function and lambda metadata
   - Registry of available functions and lambdas
   - Metadata about runtime constants and captured variables