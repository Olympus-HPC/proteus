//===-- CompilationTask.hpp -- Compilation Task header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents a particular function and specialization for compilation.
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILATION_TASK_HPP
#define PROTEUS_COMPILATION_TASK_HPP

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

/**
 * @brief Represents a compilation task with all necessary information
 * 
 * A CompilationTask captures all the information needed to compile a specific
 * function with specialized parameters. This includes the function code,
 * runtime constants, dimensions, and other configuration.
 */
class CompilationTask {
private:
  std::reference_wrapper<const Module> KernelModule;
  HashT HashValue;
  std::string KernelName;
  std::string Suffix;
  dim3 BlockDim;
  dim3 GridDim;
  SmallVector<int32_t> RCIndices;
  SmallVector<RuntimeConstant> RCVec;
  SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo;
  std::unordered_map<std::string, const void*> VarNameToDevPtr;
  SmallPtrSet<void*, 8> GlobalLinkedBinaries;
  std::string DeviceArch;
  bool UseRTC;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  bool SpecializeArgs;
  bool SpecializeDims;
  bool SpecializeLaunchBounds;

public:
  /**
   * @brief Construct a CompilationTask
   */
  CompilationTask(
      const Module& Mod, HashT HashValue, const std::string& KernelName,
      std::string& Suffix, dim3 BlockDim, dim3 GridDim,
      const SmallVector<int32_t>& RCIndices,
      const SmallVector<RuntimeConstant>& RCVec,
      const SmallVector<std::pair<std::string, StringRef>>& LambdaCalleeInfo,
      const std::unordered_map<std::string, const void*>& VarNameToDevPtr,
      const SmallPtrSet<void*, 8>& GlobalLinkedBinaries,
      const std::string& DeviceArch, bool UseRTC, bool DumpIR,
      bool RelinkGlobalsByCopy, bool SpecializeArgs, bool SpecializeDims,
      bool SpecializeLaunchBounds)
      : KernelModule(Mod), HashValue(HashValue), KernelName(KernelName),
        Suffix(Suffix), BlockDim(BlockDim), GridDim(GridDim),
        RCIndices(RCIndices), RCVec(RCVec), LambdaCalleeInfo(LambdaCalleeInfo),
        VarNameToDevPtr(VarNameToDevPtr),
        GlobalLinkedBinaries(GlobalLinkedBinaries), DeviceArch(DeviceArch),
        UseRTC(UseRTC), DumpIR(DumpIR),
        RelinkGlobalsByCopy(RelinkGlobalsByCopy),
        SpecializeArgs(SpecializeArgs), SpecializeDims(SpecializeDims),
        SpecializeLaunchBounds(SpecializeLaunchBounds) {}

  // Delete copy operations.
  CompilationTask(const CompilationTask&) = delete;
  CompilationTask& operator=(const CompilationTask&) = delete;

  // Use default move operations.
  CompilationTask(CompilationTask&&) noexcept = default;
  CompilationTask& operator=(CompilationTask&&) noexcept = default;

  /**
   * @brief Clone the module for compilation
   * 
   * Creates a new copy of the kernel module in the provided context.
   * 
   * @param Ctx LLVM context to use for the new module
   * @return Cloned module
   */
  std::unique_ptr<Module> cloneModule(LLVMContext& Ctx) const {
    SmallVector<char, 4096> ModuleStr;
    raw_svector_ostream OS(ModuleStr);
    WriteBitcodeToFile(KernelModule, OS);
    StringRef ModuleStrRef = StringRef{ModuleStr.data(), ModuleStr.size()};
    auto BufferRef = MemoryBufferRef{ModuleStrRef, ""};
    auto ClonedModule = parseBitcodeFile(BufferRef, Ctx);
    if (auto E = ClonedModule.takeError()) {
      PROTEUS_FATAL_ERROR("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::move(*ClonedModule);
  }

  /**
   * @brief Get the hash value that uniquely identifies this task
   */
  HashT getHashValue() const { return HashValue; }

  /**
   * @brief Get the kernel/function name
   */
  const std::string& getKernelName() const { return KernelName; }

  /**
   * @brief Get the mangled suffix for the kernel
   */
  const std::string& getSuffix() const { return Suffix; }

  /**
   * @brief Get the block dimensions for kernel launch
   */
  const dim3& getBlockDim() const { return BlockDim; }

  /**
   * @brief Get the grid dimensions for kernel launch
   */
  const dim3& getGridDim() const { return GridDim; }

  /**
   * @brief Get the indices of runtime constant arguments
   */
  const SmallVector<int32_t>& getRCIndices() const { return RCIndices; }

  /**
   * @brief Get the runtime constant values
   */
  const SmallVector<RuntimeConstant>& getRCValues() const { return RCVec; }

  /**
   * @brief Get lambda callee information
   */
  const SmallVector<std::pair<std::string, StringRef>>& getLambdaCalleeInfo() const {
    return LambdaCalleeInfo;
  }

  /**
   * @brief Get the map of variable names to device pointers
   */
  const std::unordered_map<std::string, const void*>& getVarNameToDevPtrMap() const {
    return VarNameToDevPtr;
  }

  /**
   * @brief Get the set of global linked binaries
   */
  const SmallPtrSet<void*, 8>& getGlobalLinkedBinaries() const {
    return GlobalLinkedBinaries;
  }

  /**
   * @brief Get the target device architecture
   */
  const std::string& getDeviceArch() const { return DeviceArch; }

  /**
   * @brief Check if RTC should be used for code generation
   */
  bool shouldUseRTC() const { return UseRTC; }

  /**
   * @brief Check if LLVM IR should be dumped during compilation
   */
  bool shouldDumpIR() const { return DumpIR; }

  /**
   * @brief Check if globals should be relinked by copy
   */
  bool shouldRelinkGlobalsByCopy() const { return RelinkGlobalsByCopy; }

  /**
   * @brief Check if arguments should be specialized
   */
  bool shouldSpecializeArgs() const { return SpecializeArgs; }

  /**
   * @brief Check if dimensions should be specialized
   */
  bool shouldSpecializeDims() const { return SpecializeDims; }

  /**
   * @brief Check if launch bounds should be specialized
   */
  bool shouldSpecializeLaunchBounds() const { return SpecializeLaunchBounds; }
};

} // namespace proteus

#endif // PROTEUS_COMPILATION_TASK_HPP