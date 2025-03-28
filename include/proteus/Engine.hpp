//===-- Engine.hpp -- JIT Engine header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Engine encapsulates the backend details and creates compilation tasks.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_ENGINE_HPP
#define PROTEUS_ENGINE_HPP

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>

#include "proteus/Builder.hpp"
#include "proteus/Cache.hpp"
#include "proteus/Code.hpp"
#include "proteus/CompilationResult.hpp"
#include "proteus/CompilationTask.hpp"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"

namespace proteus {

using namespace llvm;

/**
 * @brief Engine configuration options
 */
struct EngineConfig {
  // Feature flags
  bool UseStoredCache = false;
  bool SetLaunchBounds = true;
  bool SpecializeArgs = true;
  bool SpecializeDims = true;
  bool UseHipRtcCodegen = false;
  bool Disabled = false;
  bool DumpLlvmIr = false;
  bool RelinkGlobalsByCopy = false;
  bool AsyncCompilation = false;
  bool AsyncTestBlocking = false;
  bool UseLightweightKernelClone = true;
  
  // Configuration options
  int AsyncThreads = 4;
  CacheConfig CacheConfig;
};

/**
 * @brief Engine encapsulates backend details and creates compilation tasks
 */
class Engine {
public:
  /**
   * @brief Create an engine for the specified backend type
   */
  static std::unique_ptr<Engine> create(BackendType Backend);

  /**
   * @brief Virtual destructor
   */
  virtual ~Engine() = default;

  /**
   * @brief Get the engine configuration
   */
  const EngineConfig& getConfig() const { return Config; }

  /**
   * @brief Set the engine configuration
   */
  void setConfig(const EngineConfig& NewConfig) { Config = NewConfig; }

  /**
   * @brief Check if the engine is disabled
   */
  bool isDisabled() const { return Config.Disabled; }

  /**
   * @brief Enable the engine
   */
  void enable() { Config.Disabled = false; }

  /**
   * @brief Disable the engine
   */
  void disable() { Config.Disabled = true; }

  /**
   * @brief Create a compilation task
   * 
   * @param Code Code to compile
   * @param FunctionName Function name within the code
   * @param GridDim Grid dimensions for kernel launch
   * @param BlockDim Block dimensions for kernel launch
   * @param RuntimeConstants Runtime constants for specialization
   * @return A new compilation task
   */
  virtual std::unique_ptr<CompilationTask> createCompilationTask(
      const Code& Code,
      const std::string& FunctionName,
      const dim3& GridDim,
      const dim3& BlockDim,
      const SmallVector<RuntimeConstant>& RuntimeConstants = {}) = 0;

  /**
   * @brief Compile a task and return the result
   * 
   * @param Task Compilation task to compile
   * @return The compilation result
   */
  virtual std::unique_ptr<CompilationResult> compile(const CompilationTask& Task) = 0;

  /**
   * @brief Look up a previously compiled result in the cache
   * 
   * @param HashValue Hash of the compilation
   * @return Cached result if found, nullptr otherwise
   */
  virtual std::unique_ptr<CompilationResult> lookupCache(const HashT& HashValue) = 0;

  /**
   * @brief Register a device global variable
   * 
   * @param VarName Name of the global variable
   * @param Addr Address of the global variable
   */
  virtual void registerGlobalVariable(const char* VarName, const void* Addr) = 0;

protected:
  /**
   * @brief Construct an engine with the given configuration
   */
  explicit Engine(const EngineConfig& Config) : Config(Config) {
    // Initialize LLVM components on construction
    static InitLLVMTargets Init;
  }

  /**
   * @brief Get runtime constant values from kernel arguments
   * 
   * @param KernelArgs Kernel arguments
   * @param RCIndices Indices of runtime constants
   * @param RCTypes Types of runtime constants
   * @param RCVec Output vector for runtime constants
   */
  void getRuntimeConstantValues(void** KernelArgs,
                               const ArrayRef<int32_t> RCIndices,
                               const ArrayRef<int32_t> RCTypes,
                               SmallVector<RuntimeConstant>& RCVec);

  /**
   * @brief Optimize the LLVM IR module
   * 
   * @param M Module to optimize
   * @param Arch Target architecture
   * @param OptLevel Optimization level character ('0' to '3')
   * @param CodegenOptLevel Codegen optimization level (0 to 3)
   */
  void optimizeIR(Module& M, StringRef Arch, char OptLevel = '3',
                 unsigned CodegenOptLevel = 3);

  /**
   * @brief Run basic cleanup passes on the module
   * 
   * @param M Module to clean up
   */
  void runCleanupPassPipeline(Module& M);

  /**
   * @brief Generate a mangled suffix from a hash value
   * 
   * @param HashValue Hash value to use
   * @return Mangled suffix string
   */
  std::string mangleSuffix(HashT& HashValue);

  EngineConfig Config;
};

/**
 * @brief Engine implementation for CPU targets
 */
class CPUEngine : public Engine {
public:
  /**
   * @brief Construct a CPU engine
   */
  explicit CPUEngine(const EngineConfig& Config);

  /**
   * @brief Create a compilation task for CPU target
   */
  std::unique_ptr<CompilationTask> createCompilationTask(
      const Code& Code,
      const std::string& FunctionName,
      const dim3& GridDim,
      const dim3& BlockDim,
      const SmallVector<RuntimeConstant>& RuntimeConstants = {}) override;

  /**
   * @brief Compile a task for CPU target
   */
  std::unique_ptr<CompilationResult> compile(const CompilationTask& Task) override;

  /**
   * @brief Look up a previously compiled result in the cache
   */
  std::unique_ptr<CompilationResult> lookupCache(const HashT& HashValue) override;

  /**
   * @brief Register a global variable for CPU target
   */
  void registerGlobalVariable(const char* VarName, const void* Addr) override;

private:
  std::unique_ptr<Builder> TheBuilder;
  std::unique_ptr<Cache> TheCache;
};

/**
 * @brief Engine implementation for CUDA targets
 */
class CUDAEngine : public Engine {
public:
  /**
   * @brief Construct a CUDA engine
   */
  explicit CUDAEngine(const EngineConfig& Config);

  /**
   * @brief Create a compilation task for CUDA target
   */
  std::unique_ptr<CompilationTask> createCompilationTask(
      const Code& Code,
      const std::string& FunctionName,
      const dim3& GridDim,
      const dim3& BlockDim,
      const SmallVector<RuntimeConstant>& RuntimeConstants = {}) override;

  /**
   * @brief Compile a task for CUDA target
   */
  std::unique_ptr<CompilationResult> compile(const CompilationTask& Task) override;

  /**
   * @brief Look up a previously compiled result in the cache
   */
  std::unique_ptr<CompilationResult> lookupCache(const HashT& HashValue) override;

  /**
   * @brief Register a global variable for CUDA target
   */
  void registerGlobalVariable(const char* VarName, const void* Addr) override;

private:
  std::unique_ptr<Builder> TheBuilder;
  std::unique_ptr<Cache> TheCache;
  std::unordered_map<std::string, const void*> VarNameToDevPtr;
  std::string DeviceArch;
};

/**
 * @brief Engine implementation for HIP targets
 */
class HIPEngine : public Engine {
public:
  /**
   * @brief Construct a HIP engine
   */
  explicit HIPEngine(const EngineConfig& Config);

  /**
   * @brief Create a compilation task for HIP target
   */
  std::unique_ptr<CompilationTask> createCompilationTask(
      const Code& Code,
      const std::string& FunctionName,
      const dim3& GridDim,
      const dim3& BlockDim,
      const SmallVector<RuntimeConstant>& RuntimeConstants = {}) override;

  /**
   * @brief Compile a task for HIP target
   */
  std::unique_ptr<CompilationResult> compile(const CompilationTask& Task) override;

  /**
   * @brief Look up a previously compiled result in the cache
   */
  std::unique_ptr<CompilationResult> lookupCache(const HashT& HashValue) override;

  /**
   * @brief Register a global variable for HIP target
   */
  void registerGlobalVariable(const char* VarName, const void* Addr) override;

private:
  std::unique_ptr<Builder> TheBuilder;
  std::unique_ptr<Cache> TheCache;
  std::unordered_map<std::string, const void*> VarNameToDevPtr;
  std::string DeviceArch;
};

/**
 * @brief Factory implementation for Engine::create
 */
inline std::unique_ptr<Engine> Engine::create(BackendType Backend) {
  EngineConfig DefaultConfig;
  
  switch (Backend) {
    case BackendType::CPU:
      return std::make_unique<CPUEngine>(DefaultConfig);
    case BackendType::CUDA:
      return std::make_unique<CUDAEngine>(DefaultConfig);
    case BackendType::HIP:
      return std::make_unique<HIPEngine>(DefaultConfig);
    default:
      PROTEUS_FATAL_ERROR("Unknown backend type");
  }
}

} // namespace proteus

#endif // PROTEUS_ENGINE_HPP