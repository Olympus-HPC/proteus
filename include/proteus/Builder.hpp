//===-- Builder.hpp -- Compilation builder header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder transforms a CompilationTask into a CompilationResult.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_BUILDER_HPP
#define PROTEUS_BUILDER_HPP

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MemoryBuffer.h>

#include "proteus/CompilationTask.hpp"
#include "proteus/CompilationResult.hpp"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Logger.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

/**
 * @brief Backend type for compilation
 */
enum class BackendType {
  CPU,
  CUDA,
  HIP
};

/**
 * @brief Transforms a CompilationTask into a CompilationResult
 */
class Builder {
private:
  BackendType Backend;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  bool UseRTC;
  std::string Architecture;

public:
  /**
   * @brief Construct a Builder for the specified backend
   * 
   * @param Backend Backend to target
   * @param Architecture Target architecture
   * @param DumpIR Whether to dump the IR during compilation
   * @param RelinkGlobalsByCopy Whether to relink globals by copy
   * @param UseRTC Whether to use RTC for code generation
   */
  Builder(BackendType Backend, 
         std::string Architecture = "",
         bool DumpIR = false,
         bool RelinkGlobalsByCopy = false,
         bool UseRTC = false)
      : Backend(Backend),
        DumpIR(DumpIR),
        RelinkGlobalsByCopy(RelinkGlobalsByCopy),
        UseRTC(UseRTC),
        Architecture(std::move(Architecture)) {}

  /**
   * @brief Compile the task into a result
   * 
   * @param Task The compilation task to process
   * @return The compilation result
   */
  std::unique_ptr<CompilationResult> build(const CompilationTask& Task) {
    switch (Backend) {
      case BackendType::CPU:
        return buildForCPU(Task);
      case BackendType::CUDA:
        return buildForCUDA(Task);
      case BackendType::HIP:
        return buildForHIP(Task);
      default:
        PROTEUS_FATAL_ERROR("Unknown backend type");
    }
  }

private:
  /**
   * @brief Compile a task for CPU
   */
  std::unique_ptr<CompilationResult> buildForCPU(const CompilationTask& Task) {
    // Implemented in the future
    PROTEUS_FATAL_ERROR("CPU backend not implemented yet");
  }

  /**
   * @brief Compile a task for CUDA
   */
  std::unique_ptr<CompilationResult> buildForCUDA(const CompilationTask& Task) {
#if PROTEUS_ENABLE_DEBUG
    auto Start = std::chrono::high_resolution_clock::now();
#endif

    LLVMContext Ctx;
    std::unique_ptr<Module> M = Task.cloneModule(Ctx);

    std::string KernelMangled = (Task.getKernelName() + Task.getSuffix());

    proteus::specializeIR(*M, Task.getKernelName(), Task.getSuffix(), 
                         Task.getBlockDim(), Task.getGridDim(), 
                         Task.getRCIndices(), Task.getRCValues(),
                         Task.getLambdaCalleeInfo(), 
                         Task.shouldSpecializeArgs(),
                         Task.shouldSpecializeDims(),
                         Task.shouldSpecializeLaunchBounds());

    proteus::replaceGlobalVariablesWithPointers(*M, Task.getVarNameToDevPtrMap());

    // For CUDA, run optimization pipeline
    optimizeIR(*M, Architecture, '3', 3);

    if (DumpIR) {
      const auto CreateDumpDirectory = []() {
        const std::string DumpDirectory = ".proteus-dump";
        std::filesystem::create_directory(DumpDirectory);
        return DumpDirectory;
      };

      static const std::string DumpDirectory = CreateDumpDirectory();

      saveToFile(DumpDirectory + "/device-jit-" + Task.getHashValue().toString() + ".ll", *M);
    }

    auto ObjBuf = proteus::codegenObject(
        *M, Architecture, Task.getGlobalLinkedBinaries(), UseRTC);

    if (!RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(), Task.getVarNameToDevPtrMap());

    // In a real implementation, we'd get the actual function pointer
    void* FunctionPtr = proteus::getKernelFunctionFromImage(
        KernelMangled, ObjBuf->getBufferStart(),
        RelinkGlobalsByCopy, Task.getVarNameToDevPtrMap());

#if PROTEUS_ENABLE_DEBUG
    auto End = std::chrono::high_resolution_clock::now();
    auto Duration = End - Start;
    auto Milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(Duration).count();
    Logger::logs("proteus") << "Compiled HashValue " << Task.getHashValue().toString()
                          << " for " << Milliseconds << "ms\n";
#endif

    return std::make_unique<CompilationResult>(
        Task.getHashValue(),
        KernelMangled,
        std::move(ObjBuf),
        FunctionPtr,
        Task.getRCValues());
  }

  /**
   * @brief Compile a task for HIP
   */
  std::unique_ptr<CompilationResult> buildForHIP(const CompilationTask& Task) {
#if PROTEUS_ENABLE_DEBUG
    auto Start = std::chrono::high_resolution_clock::now();
#endif

    LLVMContext Ctx;
    std::unique_ptr<Module> M = Task.cloneModule(Ctx);

    std::string KernelMangled = (Task.getKernelName() + Task.getSuffix());

    proteus::specializeIR(*M, Task.getKernelName(), Task.getSuffix(), 
                         Task.getBlockDim(), Task.getGridDim(), 
                         Task.getRCIndices(), Task.getRCValues(),
                         Task.getLambdaCalleeInfo(), 
                         Task.shouldSpecializeArgs(),
                         Task.shouldSpecializeDims(),
                         Task.shouldSpecializeLaunchBounds());

    proteus::replaceGlobalVariablesWithPointers(*M, Task.getVarNameToDevPtrMap());

    // For HIP, only run optimization if not using RTC
    if (!UseRTC) {
      optimizeIR(*M, Architecture, '3', 3);
    }

    if (DumpIR) {
      const auto CreateDumpDirectory = []() {
        const std::string DumpDirectory = ".proteus-dump";
        std::filesystem::create_directory(DumpDirectory);
        return DumpDirectory;
      };

      static const std::string DumpDirectory = CreateDumpDirectory();

      saveToFile(DumpDirectory + "/device-jit-" + Task.getHashValue().toString() + ".ll", *M);
    }

    auto ObjBuf = proteus::codegenObject(
        *M, Architecture, Task.getGlobalLinkedBinaries(), UseRTC);

    if (!RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(), Task.getVarNameToDevPtrMap());

    // In a real implementation, we'd get the actual function pointer
    void* FunctionPtr = proteus::getKernelFunctionFromImage(
        KernelMangled, ObjBuf->getBufferStart(),
        RelinkGlobalsByCopy, Task.getVarNameToDevPtrMap());

#if PROTEUS_ENABLE_DEBUG
    auto End = std::chrono::high_resolution_clock::now();
    auto Duration = End - Start;
    auto Milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(Duration).count();
    Logger::logs("proteus") << "Compiled HashValue " << Task.getHashValue().toString()
                          << " for " << Milliseconds << "ms\n";
#endif

    return std::make_unique<CompilationResult>(
        Task.getHashValue(),
        KernelMangled,
        std::move(ObjBuf),
        FunctionPtr,
        Task.getRCValues());
  }
};

} // namespace proteus

#endif // PROTEUS_BUILDER_HPP