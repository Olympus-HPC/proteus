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
  std::reference_wrapper<const Module> KernelModule;
  HashT HashValue;
  std::string KernelName;
  std::string Suffix;
  dim3 BlockDim;
  dim3 GridDim;
  SmallVector<int32_t> RCIndices;
  SmallVector<RuntimeConstant> RCVec;
  SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  std::string DeviceArch;
  bool UseRTC;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  bool SpecializeArgs;
  bool SpecializeDims;
  bool SpecializeLaunchBounds;

  std::unique_ptr<Module> cloneKernelModule(LLVMContext &Ctx) {
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

public:
  CompilationTask(
      const Module &Mod, HashT HashValue, const std::string &KernelName,
      std::string &Suffix, dim3 BlockDim, dim3 GridDim,
      const SmallVector<int32_t> &RCIndices,
      const SmallVector<RuntimeConstant> &RCVec,
      const SmallVector<std::pair<std::string, StringRef>> &LambdaCalleeInfo,
      const std::unordered_map<std::string, const void *> &VarNameToDevPtr,
      const SmallPtrSet<void *, 8> &GlobalLinkedBinaries,
      const std::string &DeviceArch, bool UseRTC, bool DumpIR,
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

    proteus::specializeIR(*M, KernelName, Suffix, BlockDim, GridDim, RCIndices,
                          RCVec, LambdaCalleeInfo, SpecializeArgs,
                          SpecializeDims, SpecializeLaunchBounds);

    replaceGlobalVariablesWithPointers(*M, VarNameToDevPtr);

    // For HIP RTC codegen do not run the optimization pipeline since HIP
    // RTC internally runs it. For the rest of cases, that is CUDA or HIP
    // with our own codegen instead of RTC, run the target-specific
    // optimization pipeline to optimize the LLVM IR before handing over
    // to codegen.
#if PROTEUS_ENABLE_CUDA
    optimizeIR(*M, DeviceArch, "default<O3>", 3);
#elif PROTEUS_ENABLE_HIP
    if (!UseRTC)
      optimizeIR(*M, DeviceArch, "default<O3>", 3);
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
        proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries, UseRTC);

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
