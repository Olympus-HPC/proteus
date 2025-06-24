#ifndef PROTEUS_COMPILATION_TASK_HPP
#define PROTEUS_COMPILATION_TASK_HPP

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

class CompilationTask {
private:
  MemoryBufferRef Bitcode;
  HashT HashValue;
  std::string KernelName;
  std::string Suffix;
  dim3 BlockDim;
  dim3 GridDim;
  SmallVector<RuntimeConstant> RCVec;
  SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  std::string DeviceArch;
  CodegenOption CGOption;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  bool SpecializeArgs;
  bool SpecializeDims;
  bool SpecializeLaunchBounds;

  std::unique_ptr<Module> cloneKernelModule(LLVMContext &Ctx) {
    auto ClonedModule = parseBitcodeFile(Bitcode, Ctx);
    if (auto E = ClonedModule.takeError()) {
      PROTEUS_FATAL_ERROR("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::move(*ClonedModule);
  }

public:
  CompilationTask(
      MemoryBufferRef Bitcode, HashT HashValue, const std::string &KernelName,
      std::string &Suffix, dim3 BlockDim, dim3 GridDim,
      const SmallVector<RuntimeConstant> &RCVec,
      const SmallVector<std::pair<std::string, StringRef>> &LambdaCalleeInfo,
      const std::unordered_map<std::string, const void *> &VarNameToDevPtr,
      const SmallPtrSet<void *, 8> &GlobalLinkedBinaries,
      const std::string &DeviceArch, CodegenOption CGOption, bool DumpIR,
      bool RelinkGlobalsByCopy, bool SpecializeArgs, bool SpecializeDims,
      bool SpecializeLaunchBounds)
      : Bitcode(Bitcode), HashValue(HashValue), KernelName(KernelName),
        Suffix(Suffix), BlockDim(BlockDim), GridDim(GridDim), RCVec(RCVec),
        LambdaCalleeInfo(LambdaCalleeInfo), VarNameToDevPtr(VarNameToDevPtr),
        GlobalLinkedBinaries(GlobalLinkedBinaries), DeviceArch(DeviceArch),
        CGOption(CGOption), DumpIR(DumpIR),
        RelinkGlobalsByCopy(RelinkGlobalsByCopy),
        SpecializeArgs(SpecializeArgs), SpecializeDims(SpecializeDims),
        SpecializeLaunchBounds(SpecializeLaunchBounds) {}

  // Delete copy operations.
  CompilationTask(const CompilationTask &) = delete;
  CompilationTask &operator=(const CompilationTask &) = delete;

  // Use default move operations.
  CompilationTask(CompilationTask &&) noexcept = default;
  CompilationTask &operator=(CompilationTask &&) noexcept = default;

  HashT getHashValue() const { return HashValue; }

  std::unique_ptr<MemoryBuffer> compile() {
#if PROTEUS_ENABLE_DEBUG
    auto Start = std::chrono::high_resolution_clock::now();
#endif

    LLVMContext Ctx;
    std::unique_ptr<Module> M = cloneKernelModule(Ctx);

    std::string KernelMangled = (KernelName + Suffix);

    proteus::specializeIR(*M, KernelName, Suffix, BlockDim, GridDim, RCVec,
                          LambdaCalleeInfo, SpecializeArgs, SpecializeDims,
                          SpecializeLaunchBounds);

    replaceGlobalVariablesWithPointers(*M, VarNameToDevPtr);

#if PROTEUS_ENABLE_CUDA
    // For CUDA we always run the optimization pipeline.
    optimizeIR(*M, DeviceArch, '3', 3);
#elif PROTEUS_ENABLE_HIP
    // For HIP RTC codegen we run the optimization pipeline only for Serial and
    // Parallel codegen since those do not run it internally. HIP RTC and
    // Parallel ThinLTO invoke optimization internally.
    // TODO: Move optimizeIR inside the codegen routines?
    if (CGOption == CodegenOption::Serial ||
        CGOption == CodegenOption::Parallel)
      optimizeIR(*M, DeviceArch, '3', 3);
#else
#error "JitEngineDevice requires PROTEUS_ENABLE_CUDA or PROTEUS_ENABLE_HIP"
#endif

    if (DumpIR) {
      const auto CreateDumpDirectory = []() {
        const std::string DumpDirectory = ".proteus-dump";
        std::filesystem::create_directory(DumpDirectory);
        return DumpDirectory;
      };

      static const std::string DumpDirectory = CreateDumpDirectory();

      saveToFile(DumpDirectory + "/device-jit-" + HashValue.toString() + ".ll",
                 *M);
    }

    auto ObjBuf =
        proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries, CGOption);

    if (!RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(), VarNameToDevPtr);

#if PROTEUS_ENABLE_DEBUG
    auto End = std::chrono::high_resolution_clock::now();
    auto Duration = End - Start;
    auto Milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(Duration).count();
    Logger::logs("proteus") << "Compiled HashValue " << HashValue.toString()
                            << " for " << Milliseconds << "ms\n";
#endif

    return ObjBuf;
  }
};

} // namespace proteus

#endif
