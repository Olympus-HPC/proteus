# Proteus Design Architecture

This document describes the high-level architecture of the Proteus JIT compilation system.

## System Overview

Proteus is a JIT (Just-In-Time) compilation framework that specializes code at runtime based on runtime constants and execution dimensions. The system is designed to work across different backends (CPU, CUDA, HIP) while maintaining a clean separation of concerns.

The refactored architecture is based on the following key components:

1. **Engine**: Encapsulates backend details and creates compilation tasks
2. **CompilationTask**: Represents a function with specific specialization parameters
3. **Builder**: Transforms a CompilationTask into a CompilationResult
4. **CompilationResult**: Contains function pointer and necessary execution information
5. **Cache**: Manages storage and retrieval of compilation results
6. **Code**: Represents the extracted function code (LLVM IR)
7. **CodeContext**: Maintains a registry of functions and lambdas with metadata

## Component Interfaces

### Engine

The Engine component encapsulates all backend-specific details and creates compilation tasks. It handles configuration settings and initiates the compilation process.

```cpp
class Engine {
public:
  static std::unique_ptr<Engine> create(BackendType Backend);
  virtual ~Engine() = default;

  const EngineConfig& getConfig() const;
  void setConfig(const EngineConfig& NewConfig);
  bool isDisabled() const;
  void enable();
  void disable();

  virtual std::unique_ptr<CompilationTask> createCompilationTask(
      const Code& Code,
      const std::string& FunctionName,
      const dim3& GridDim,
      const dim3& BlockDim,
      const SmallVector<RuntimeConstant>& RuntimeConstants = {}) = 0;
      
  virtual std::unique_ptr<CompilationResult> compile(const CompilationTask& Task) = 0;
  virtual std::unique_ptr<CompilationResult> lookupCache(const HashT& HashValue) = 0;
  virtual void registerGlobalVariable(const char* VarName, const void* Addr) = 0;
  
protected:
  void getRuntimeConstantValues(void** KernelArgs,
                               const ArrayRef<int32_t> RCIndices,
                               const ArrayRef<int32_t> RCTypes,
                               SmallVector<RuntimeConstant>& RCVec);
                               
  void optimizeIR(Module& M, StringRef Arch, char OptLevel, unsigned CodegenOptLevel);
  void runCleanupPassPipeline(Module& M);
  std::string mangleSuffix(HashT& HashValue);
};
```

Engine implementations:
- **CPUEngine**: Implementation for CPU targets using LLVM JIT compiler
- **CUDAEngine**: Implementation for NVIDIA GPUs using NVPTX backend
- **HIPEngine**: Implementation for AMD GPUs using ROCm/HIP backend

Each backend implementation handles:
- Target-specific compilation strategies
- Hardware-specific optimizations and code generation
- Device-specific resource management
- Global variable handling in device memory

### CompilationTask

The CompilationTask represents a particular function and specialization. It contains all the information needed for compilation, including runtime constants, dimensions, and configuration flags.

```cpp
class CompilationTask {
public:
  CompilationTask(const Module& Mod, HashT HashValue, const std::string& KernelName,
                 /* other parameters */);

  std::unique_ptr<Module> cloneModule(LLVMContext& Ctx) const;
  HashT getHashValue() const;
  const std::string& getKernelName() const;
  /* other accessors */
};
```

### Builder

The Builder component handles the transformation of a CompilationTask into a CompilationResult. It encapsulates the details of the compilation process.

```cpp
class Builder {
public:
  virtual ~Builder() = default;
  virtual std::unique_ptr<CompilationResult> build(const CompilationTask& Task) = 0;
};
```

Builder implementations:
- **SyncBuilder**: Synchronous compilation in the current thread
  ```cpp
  class SyncBuilder : public Builder {
  public:
    static SyncBuilder& instance();
    std::unique_ptr<CompilationResult> build(const CompilationTask& Task) override;
  private:
    SyncBuilder() = default;
  };
  ```

- **AsyncBuilder**: Asynchronous compilation using a thread pool
  ```cpp
  class AsyncBuilder : public Builder {
  public:
    static AsyncBuilder& instance(int NumThreads = 4);
    std::unique_ptr<CompilationResult> build(const CompilationTask& Task) override;
    bool isCompilationPending(const HashT& HashValue);
    std::unique_ptr<CompilationResult> getResult(const HashT& HashValue, bool BlockingWait = false);
    void joinAllThreads();
  private:
    // Thread pool and task queue management
  };
  ```

Each Builder implementation is responsible for:
- Cloning and preparing the IR module
- Applying transformations (specialization, inlining, etc.)
- Optimizing the code for the target architecture
- Generating machine code
- Creating a CompilationResult with the function pointer

### CompilationResult

The CompilationResult contains the function pointer and all associated metadata needed to execute a JIT-compiled function.

```cpp
class CompilationResult {
public:
  CompilationResult(HashT HashValue, std::string MangledName,
                  std::unique_ptr<MemoryBuffer> ObjBuffer,
                  void* FunctionPtr,
                  const SmallVector<RuntimeConstant>& RuntimeConstants);

  HashT getHashValue() const;
  const std::string& getMangledName() const;
  const MemoryBuffer& getObjectBuffer() const;
  std::unique_ptr<MemoryBuffer> takeObjectBuffer();
  
  template<typename FuncType>
  FuncType getFunction() const;
  
  void* getFunctionPtr() const;
  const SmallVector<RuntimeConstant>& getRuntimeConstants() const;
};
```

### Cache

The Cache component handles storage and retrieval of compilation results. Multiple implementations provide in-memory caching, disk-based persistence, or combinations.

```cpp
class Cache {
public:
  static std::unique_ptr<Cache> create(const CacheConfig& Config);
  virtual ~Cache() = default;

  virtual std::unique_ptr<CompilationResult> lookup(const HashT& HashValue) = 0;
  virtual void store(std::unique_ptr<CompilationResult> Result) = 0;
  virtual void printStats() const = 0;
};
```

Cache implementations:
- MemoryCache
- DiskCache
- HierarchicalCache

### Code

The Code component represents the extracted representation of a function that will be used in compilation tasks. In the current implementation, this is LLVM IR.

```cpp
class Code {
public:
  Code(std::unique_ptr<llvm::Module> Module, std::string FunctionName);
  
  const std::string& getFunctionName() const;
  llvm::Module& getModule();
  const llvm::Module& getModule() const;
  std::unique_ptr<llvm::Module> takeModule();
  
  std::unique_ptr<Code> clone(LLVMContext& Ctx) const;
  void markOptimized();
  bool isOptimized() const;
  HashT getHash() const;
};
```

### CodeContext

The CodeContext maintains a registry of functions and lambdas available for JIT compilation, along with their metadata such as runtime constant argument indices, types, and capture values.

```cpp
class CodeContext {
public:
  static CodeContext& instance();
  
  void registerFunction(StringRef Name, void* FuncPtr, const int32_t* RCIndices,
                       const int32_t* RCTypes, int32_t NumRCs);
  
  std::optional<std::reference_wrapper<const FunctionInfo>> 
  lookupFunction(StringRef Name) const;
  
  void pushJitVariable(RuntimeConstant &RC);
  void registerLambda(const char *LambdaType);
  const SmallVector<RuntimeConstant>& getJitVariables(StringRef LambdaTypeRef);
  
  std::optional<std::reference_wrapper<const LambdaInfo>>
  matchLambda(StringRef FnName) const;
  
  bool empty() const;
};
```

## Component Relationships

The relationships between components form a clear chain of responsibility:

1. **Engine** is the core component that creates and manages **CompilationTasks**
2. **CompilationTask** captures all specialization parameters for a function
3. **Builder** transforms a **CompilationTask** into a **CompilationResult**
4. **CompilationResult** contains the executable function and metadata
5. **Cache** provides storage and retrieval for **CompilationResults**
6. **Code** represents the source code that will be specialized
7. **CodeContext** keeps track of function and lambda metadata

[Component Diagram](docs/assets/component-diagram.md)

## Use Flow

1. User code calls into Proteus requesting JIT compilation of a function
2. The system obtains or creates an appropriate Engine instance for the target backend
3. The Engine creates a Code object that encapsulates the function's IR
4. The Engine creates a CompilationTask with the Code, function name, runtime constants, etc.
5. The Engine computes a hash value for the CompilationTask and checks the Cache
6. If a cached CompilationResult is found, it is returned immediately
7. Otherwise, the Engine delegates to a Builder (synchronous or asynchronous)
8. The Builder:
   - Clones and prepares the module from the CompilationTask
   - Specializes the IR using runtime constants and launch dimensions
   - Optimizes the IR for the target architecture
   - Generates object code (machine code, PTX, or HSACO)
   - Loads the object code and extracts the function pointer
   - Returns a CompilationResult containing the function pointer and metadata
9. The Engine stores the CompilationResult in the Cache for future reuse
10. The Engine returns the CompilationResult to the caller
11. The caller extracts the function pointer and executes the compiled code

## Migration Strategy

To facilitate the transition from the old architecture to the new one, we've implemented:

1. **Feature flags**: `USE_NEW_CACHE` and `USE_NEW_ENGINE` control whether to use the new components
2. **Adapters**: `JitCacheAdapter` provides a bridge between old and new cache implementations
3. **Dual implementation**: Methods like `compileAndLink` support both old and new approaches
4. **Gradual refactoring**: Starting with the CPU backend and progressively moving to CUDA and HIP
5. **Backward compatibility**: Ensuring existing code continues to work during the transition

## Benefits of this Architecture

1. **Separation of Concerns**: Each component has a clear, focused responsibility
2. **Modularity**: Components can be enhanced or replaced independently
3. **Extensibility**: New backend types or caching strategies can be added with minimal changes
4. **Testability**: Components can be tested in isolation
5. **Maintainability**: Clean interfaces reduce coupling between components