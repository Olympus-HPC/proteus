//===-- Engine.cpp -- JIT Engine implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Engine implementations for CPU, CUDA, and HIP targets.
//
//===----------------------------------------------------------------------===//

#include "proteus/Engine.hpp"
#include "proteus/AsyncBuilder.hpp"
#include "proteus/SyncBuilder.hpp"
#include "proteus/TimeTracing.hpp"

#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Target/TargetMachine.h>

namespace proteus {

// Engine base class implementation methods

void Engine::getRuntimeConstantValues(void** KernelArgs,
                                     const ArrayRef<int32_t> RCIndices,
                                     const ArrayRef<int32_t> RCTypes,
                                     SmallVector<RuntimeConstant>& RCVec) {
  // Extract runtime constants from kernel arguments
  for (size_t i = 0; i < RCIndices.size(); i++) {
    int32_t Idx = RCIndices[i];
    int32_t Type = RCTypes[i];
    void* ArgPtr = KernelArgs[Idx];
    
    RuntimeConstant RC;
    RC.ArgIdx = Idx;
    RC.Type = Type;
    
    switch (Type) {
      case PROTEUS_I32: RC.Value.Int32 = *static_cast<int32_t*>(ArgPtr); break;
      case PROTEUS_I64: RC.Value.Int64 = *static_cast<int64_t*>(ArgPtr); break;
      case PROTEUS_F32: RC.Value.Float = *static_cast<float*>(ArgPtr); break;
      case PROTEUS_F64: RC.Value.Double = *static_cast<double*>(ArgPtr); break;
      default: PROTEUS_FATAL_ERROR("Unknown runtime constant type");
    }
    
    RCVec.push_back(RC);
  }
}

void Engine::optimizeIR(Module& M, StringRef Arch, char OptLevel, unsigned CodegenOptLevel) {
  TIMESCOPE("Engine::optimizeIR");
  
  // Set up the optimization pipeline similar to -O3
  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&M);

  // Add target-specific info
  auto TLII = std::make_unique<TargetLibraryInfoImpl>(Triple(Arch));
  PM.add(new TargetLibraryInfoWrapperPass(*TLII));

  // Add standard optimization passes
  PassManagerBuilder PMB;
  PMB.OptLevel = OptLevel - '0';
  PMB.SizeLevel = 0;
  PMB.Inliner = createFunctionInliningPass(PMB.OptLevel, PMB.SizeLevel, false);
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;

  PMB.populateFunctionPassManager(FPM);
  PMB.populateModulePassManager(PM);

  // Run optimizations
  FPM.doInitialization();
  for (auto &F : M) {
    FPM.run(F);
  }
  FPM.doFinalization();
  PM.run(M);
}

void Engine::runCleanupPassPipeline(Module& M) {
  TIMESCOPE("Engine::runCleanupPassPipeline");
  
  // Apply simple cleanup passes
  legacy::PassManager PM;
  
  // Add cleanup passes
  PM.add(createInternalizePass());
  PM.add(createGlobalDCEPass());
  PM.add(createStripDeadPrototypesPass());
  
  PM.run(M);
}

std::string Engine::mangleSuffix(HashT& HashValue) {
  return "_" + HashValue.toString().substr(0, 6);
}

// CPUEngine implementation

CPUEngine::CPUEngine(const EngineConfig& Config) : Engine(Config) {
  // Create the appropriate builder based on configuration
  if (Config.AsyncCompilation) {
    TheBuilder = std::unique_ptr<Builder>(
        &AsyncBuilder::instance(Config.AsyncThreads));
  } else {
    TheBuilder = std::unique_ptr<Builder>(&SyncBuilder::instance());
  }
  
  // Create the cache
  TheCache = Cache::create(Config.CacheConfig);
}

std::unique_ptr<CompilationTask> CPUEngine::createCompilationTask(
    const Code& Code,
    const std::string& FunctionName,
    const dim3& GridDim,
    const dim3& BlockDim,
    const SmallVector<RuntimeConstant>& RuntimeConstants) {
  
  TIMESCOPE("CPUEngine::createCompilationTask");
  
  // Generate a hash for the compilation
  HashT HashValue = Code.getHash();
  
  // For CPU targets, we don't need grid/block dimensions in the same way as GPU,
  // but we include them in the hash for consistency and potential parallel CPU implementations
  std::string GridDimStr = std::to_string(GridDim.x) + "x" + 
                          std::to_string(GridDim.y) + "x" + 
                          std::to_string(GridDim.z);
  std::string BlockDimStr = std::to_string(BlockDim.x) + "x" + 
                           std::to_string(BlockDim.y) + "x" + 
                           std::to_string(BlockDim.z);
  
  HashValue = hash_combine(HashValue, hash(GridDimStr));
  HashValue = hash_combine(HashValue, hash(BlockDimStr));
  
  // Include runtime constants in the hash
  for (const auto& RC : RuntimeConstants) {
    HashValue = hash_combine(HashValue, hash(RC.ArgIdx));
    HashValue = hash_combine(HashValue, hash(RC.Type));
    
    switch (RC.Type) {
      case PROTEUS_I32: HashValue = hash_combine(HashValue, hash(RC.Value.Int32)); break;
      case PROTEUS_I64: HashValue = hash_combine(HashValue, hash(RC.Value.Int64)); break;
      case PROTEUS_F32: HashValue = hash_combine(HashValue, hash(RC.Value.Float)); break;
      case PROTEUS_F64: HashValue = hash_combine(HashValue, hash(RC.Value.Double)); break;
      default: PROTEUS_FATAL_ERROR("Unknown runtime constant type");
    }
  }
  
  // Generate a suffix for the kernel name
  std::string Suffix = mangleSuffix(HashValue);
  
  // CPU-specific indices and types for runtime constants
  SmallVector<int32_t> RCIndices;
  for (size_t i = 0; i < RuntimeConstants.size(); i++) {
    RCIndices.push_back(RuntimeConstants[i].ArgIdx);
  }
  
  // Create the task
  return std::make_unique<CompilationTask>(
      /* Mod */ Code.getModule(),
      /* HashValue */ HashValue,
      /* KernelName */ FunctionName,
      /* Suffix */ Suffix,
      /* BlockDim */ BlockDim,
      /* GridDim */ GridDim,
      /* RCIndices */ RCIndices,
      /* RCVec */ RuntimeConstants,
      /* LambdaCalleeInfo */ SmallVector<std::pair<std::string, StringRef>>{},
      /* VarNameToDevPtr */ std::unordered_map<std::string, const void*>{},
      /* GlobalLinkedBinaries */ SmallPtrSet<void*, 8>{},
      /* DeviceArch */ "x86_64",
      /* UseRTC */ false,
      /* DumpIR */ Config.DumpLlvmIr,
      /* RelinkGlobalsByCopy */ Config.RelinkGlobalsByCopy,
      /* SpecializeArgs */ Config.SpecializeArgs,
      /* SpecializeDims */ Config.SpecializeDims,
      /* SpecializeLaunchBounds */ Config.SetLaunchBounds
  );
}

std::unique_ptr<CompilationResult> CPUEngine::compile(const CompilationTask& Task) {
  TIMESCOPE("CPUEngine::compile");
  
  if (isDisabled()) {
    PROTEUS_FATAL_ERROR("Engine is disabled");
  }
  
  // Check cache first
  auto Result = lookupCache(Task.getHashValue());
  if (Result) {
    return Result;
  }
  
  // Delegate to the builder
  Result = TheBuilder->build(Task);
  
  // Store in cache
  TheCache->store(std::make_unique<CompilationResult>(
      Result->getHashValue(),
      Result->getMangledName(),
      Result->takeObjectBuffer(),
      Result->getFunctionPtr(),
      Result->getRuntimeConstants()));
  
  return Result;
}

std::unique_ptr<CompilationResult> CPUEngine::lookupCache(const HashT& HashValue) {
  TIMESCOPE("CPUEngine::lookupCache");
  
  return TheCache->lookup(HashValue);
}

void CPUEngine::registerGlobalVariable(const char* VarName, const void* Addr) {
  // CPU global variables don't need special handling in this implementation
  // In a real implementation, this would register the variable with the JIT engine
}

// CUDAEngine implementation

CUDAEngine::CUDAEngine(const EngineConfig& Config) : Engine(Config) {
  // Create the appropriate builder based on configuration
  if (Config.AsyncCompilation) {
    TheBuilder = std::unique_ptr<Builder>(
        &AsyncBuilder::instance(Config.AsyncThreads));
  } else {
    TheBuilder = std::unique_ptr<Builder>(&SyncBuilder::instance());
  }
  
  // Create the cache
  TheCache = Cache::create(Config.CacheConfig);
  
  // Get the CUDA device architecture
  // In a real implementation, we would detect the device architecture
  DeviceArch = "sm_70";
}

std::unique_ptr<CompilationTask> CUDAEngine::createCompilationTask(
    const Code& Code,
    const std::string& FunctionName,
    const dim3& GridDim,
    const dim3& BlockDim,
    const SmallVector<RuntimeConstant>& RuntimeConstants) {
  
  TIMESCOPE("CUDAEngine::createCompilationTask");
  
  // Generate a hash for the compilation
  HashT HashValue = Code.getHash();
  
  // Include grid and block dimensions in the hash
  std::string GridDimStr = std::to_string(GridDim.x) + "x" + 
                          std::to_string(GridDim.y) + "x" + 
                          std::to_string(GridDim.z);
  std::string BlockDimStr = std::to_string(BlockDim.x) + "x" + 
                           std::to_string(BlockDim.y) + "x" + 
                           std::to_string(BlockDim.z);
  
  HashValue = hash_combine(HashValue, hash(GridDimStr));
  HashValue = hash_combine(HashValue, hash(BlockDimStr));
  
  // Include runtime constants in the hash
  for (const auto& RC : RuntimeConstants) {
    HashValue = hash_combine(HashValue, hash(RC.ArgIdx));
    HashValue = hash_combine(HashValue, hash(RC.Type));
    
    switch (RC.Type) {
      case PROTEUS_I32: HashValue = hash_combine(HashValue, hash(RC.Value.Int32)); break;
      case PROTEUS_I64: HashValue = hash_combine(HashValue, hash(RC.Value.Int64)); break;
      case PROTEUS_F32: HashValue = hash_combine(HashValue, hash(RC.Value.Float)); break;
      case PROTEUS_F64: HashValue = hash_combine(HashValue, hash(RC.Value.Double)); break;
      default: PROTEUS_FATAL_ERROR("Unknown runtime constant type");
    }
  }
  
  // Generate a suffix for the kernel name
  std::string Suffix = mangleSuffix(HashValue);
  
  // CUDA-specific indices and types for runtime constants
  SmallVector<int32_t> RCIndices;
  for (size_t i = 0; i < RuntimeConstants.size(); i++) {
    RCIndices.push_back(RuntimeConstants[i].ArgIdx);
  }
  
  // Create the task
  return std::make_unique<CompilationTask>(
      /* Mod */ Code.getModule(),
      /* HashValue */ HashValue,
      /* KernelName */ FunctionName,
      /* Suffix */ Suffix,
      /* BlockDim */ BlockDim,
      /* GridDim */ GridDim,
      /* RCIndices */ RCIndices,
      /* RCVec */ RuntimeConstants,
      /* LambdaCalleeInfo */ SmallVector<std::pair<std::string, StringRef>>{},
      /* VarNameToDevPtr */ VarNameToDevPtr,
      /* GlobalLinkedBinaries */ SmallPtrSet<void*, 8>{},
      /* DeviceArch */ DeviceArch,
      /* UseRTC */ Config.UseHipRtcCodegen,
      /* DumpIR */ Config.DumpLlvmIr,
      /* RelinkGlobalsByCopy */ Config.RelinkGlobalsByCopy,
      /* SpecializeArgs */ Config.SpecializeArgs,
      /* SpecializeDims */ Config.SpecializeDims,
      /* SpecializeLaunchBounds */ Config.SetLaunchBounds
  );
}

std::unique_ptr<CompilationResult> CUDAEngine::compile(const CompilationTask& Task) {
  TIMESCOPE("CUDAEngine::compile");
  
  if (isDisabled()) {
    PROTEUS_FATAL_ERROR("Engine is disabled");
  }
  
  // Check cache first
  auto Result = lookupCache(Task.getHashValue());
  if (Result) {
    return Result;
  }
  
  // Delegate to the builder
  Result = TheBuilder->build(Task);
  
  // Store in cache
  TheCache->store(std::make_unique<CompilationResult>(
      Result->getHashValue(),
      Result->getMangledName(),
      Result->takeObjectBuffer(),
      Result->getFunctionPtr(),
      Result->getRuntimeConstants()));
  
  return Result;
}

std::unique_ptr<CompilationResult> CUDAEngine::lookupCache(const HashT& HashValue) {
  TIMESCOPE("CUDAEngine::lookupCache");
  
  return TheCache->lookup(HashValue);
}

void CUDAEngine::registerGlobalVariable(const char* VarName, const void* Addr) {
  VarNameToDevPtr[VarName] = Addr;
}

// HIPEngine implementation

HIPEngine::HIPEngine(const EngineConfig& Config) : Engine(Config) {
  // Create the appropriate builder based on configuration
  if (Config.AsyncCompilation) {
    TheBuilder = std::unique_ptr<Builder>(
        &AsyncBuilder::instance(Config.AsyncThreads));
  } else {
    TheBuilder = std::unique_ptr<Builder>(&SyncBuilder::instance());
  }
  
  // Create the cache
  TheCache = Cache::create(Config.CacheConfig);
  
  // Get the HIP device architecture
  // In a real implementation, we would detect the device architecture
  DeviceArch = "gfx906";
}

std::unique_ptr<CompilationTask> HIPEngine::createCompilationTask(
    const Code& Code,
    const std::string& FunctionName,
    const dim3& GridDim,
    const dim3& BlockDim,
    const SmallVector<RuntimeConstant>& RuntimeConstants) {
  
  TIMESCOPE("HIPEngine::createCompilationTask");
  
  // Generate a hash for the compilation
  HashT HashValue = Code.getHash();
  
  // Include grid and block dimensions in the hash
  std::string GridDimStr = std::to_string(GridDim.x) + "x" + 
                          std::to_string(GridDim.y) + "x" + 
                          std::to_string(GridDim.z);
  std::string BlockDimStr = std::to_string(BlockDim.x) + "x" + 
                           std::to_string(BlockDim.y) + "x" + 
                           std::to_string(BlockDim.z);
  
  HashValue = hash_combine(HashValue, hash(GridDimStr));
  HashValue = hash_combine(HashValue, hash(BlockDimStr));
  
  // Include runtime constants in the hash
  for (const auto& RC : RuntimeConstants) {
    HashValue = hash_combine(HashValue, hash(RC.ArgIdx));
    HashValue = hash_combine(HashValue, hash(RC.Type));
    
    switch (RC.Type) {
      case PROTEUS_I32: HashValue = hash_combine(HashValue, hash(RC.Value.Int32)); break;
      case PROTEUS_I64: HashValue = hash_combine(HashValue, hash(RC.Value.Int64)); break;
      case PROTEUS_F32: HashValue = hash_combine(HashValue, hash(RC.Value.Float)); break;
      case PROTEUS_F64: HashValue = hash_combine(HashValue, hash(RC.Value.Double)); break;
      default: PROTEUS_FATAL_ERROR("Unknown runtime constant type");
    }
  }
  
  // Generate a suffix for the kernel name
  std::string Suffix = mangleSuffix(HashValue);
  
  // HIP-specific indices and types for runtime constants
  SmallVector<int32_t> RCIndices;
  for (size_t i = 0; i < RuntimeConstants.size(); i++) {
    RCIndices.push_back(RuntimeConstants[i].ArgIdx);
  }
  
  // Create the task
  return std::make_unique<CompilationTask>(
      /* Mod */ Code.getModule(),
      /* HashValue */ HashValue,
      /* KernelName */ FunctionName,
      /* Suffix */ Suffix,
      /* BlockDim */ BlockDim,
      /* GridDim */ GridDim,
      /* RCIndices */ RCIndices,
      /* RCVec */ RuntimeConstants,
      /* LambdaCalleeInfo */ SmallVector<std::pair<std::string, StringRef>>{},
      /* VarNameToDevPtr */ VarNameToDevPtr,
      /* GlobalLinkedBinaries */ SmallPtrSet<void*, 8>{},
      /* DeviceArch */ DeviceArch,
      /* UseRTC */ Config.UseHipRtcCodegen,
      /* DumpIR */ Config.DumpLlvmIr,
      /* RelinkGlobalsByCopy */ Config.RelinkGlobalsByCopy,
      /* SpecializeArgs */ Config.SpecializeArgs,
      /* SpecializeDims */ Config.SpecializeDims,
      /* SpecializeLaunchBounds */ Config.SetLaunchBounds
  );
}

std::unique_ptr<CompilationResult> HIPEngine::compile(const CompilationTask& Task) {
  TIMESCOPE("HIPEngine::compile");
  
  if (isDisabled()) {
    PROTEUS_FATAL_ERROR("Engine is disabled");
  }
  
  // Check cache first
  auto Result = lookupCache(Task.getHashValue());
  if (Result) {
    return Result;
  }
  
  // Delegate to the builder
  Result = TheBuilder->build(Task);
  
  // Store in cache
  TheCache->store(std::make_unique<CompilationResult>(
      Result->getHashValue(),
      Result->getMangledName(),
      Result->takeObjectBuffer(),
      Result->getFunctionPtr(),
      Result->getRuntimeConstants()));
  
  return Result;
}

std::unique_ptr<CompilationResult> HIPEngine::lookupCache(const HashT& HashValue) {
  TIMESCOPE("HIPEngine::lookupCache");
  
  return TheCache->lookup(HashValue);
}

void HIPEngine::registerGlobalVariable(const char* VarName, const void* Addr) {
  VarNameToDevPtr[VarName] = Addr;
}

} // namespace proteus