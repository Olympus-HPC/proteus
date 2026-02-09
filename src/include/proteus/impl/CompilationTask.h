#ifndef PROTEUS_COMPILATION_TASK_H
#define PROTEUS_COMPILATION_TASK_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Config.h"
#include "proteus/CoreLLVM.h"
#include "proteus/CoreLLVMDevice.h"
#include "proteus/Debug.h"
#include "proteus/Hashing.h"
#include "proteus/Utils.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

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
  std::unordered_map<std::string, GlobalVarInfo> VarNameToGlobalInfo;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;
  std::string DeviceArch;
  CodegenOption CGOption;
  bool DumpIR;
  bool RelinkGlobalsByCopy;
  int MinBlocksPerSM;
  bool SpecializeArgs;
  bool SpecializeDims;
  bool SpecializeDimsRange;
  bool SpecializeLaunchBounds;
  char OptLevel;
  unsigned CodegenOptLevel;
  std::optional<std::string> PassPipeline;

  std::unique_ptr<Module> cloneKernelModule(LLVMContext &Ctx) {
    auto ClonedModule = parseBitcodeFile(Bitcode, Ctx);
    if (auto E = ClonedModule.takeError()) {
      reportFatalError("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::move(*ClonedModule);
  }

  void invokeOptimizeIR(Module &M) {
#if PROTEUS_ENABLE_CUDA
    // For CUDA we always run the optimization pipeline.
    if (!PassPipeline)
      optimizeIR(M, DeviceArch, OptLevel, CodegenOptLevel);
    else
      optimizeIR(M, DeviceArch, PassPipeline.value(), CodegenOptLevel);
#elif PROTEUS_ENABLE_HIP
    // For HIP we run the optimization pipeline only for Serial codegen. HIP RTC
    // and Parallel codegen, which uses LTO, invoke optimization internally.
    // TODO: Move optimizeIR inside the codegen routines?
    if (CGOption == CodegenOption::Serial) {
      if (!PassPipeline) {
        optimizeIR(M, DeviceArch, OptLevel, CodegenOptLevel);
      } else {
        optimizeIR(M, DeviceArch, PassPipeline.value(), CodegenOptLevel);
      }
    }
#else
#error "JitEngineDevice requires PROTEUS_ENABLE_CUDA or PROTEUS_ENABLE_HIP"
#endif
  }

public:
  CompilationTask(
      MemoryBufferRef Bitcode, HashT HashValue, const std::string &KernelName,
      std::string &Suffix, dim3 BlockDim, dim3 GridDim,
      const SmallVector<RuntimeConstant> &RCVec,
      const SmallVector<std::pair<std::string, StringRef>> &LambdaCalleeInfo,
      const std::unordered_map<std::string, GlobalVarInfo> &VarNameToGlobalInfo,
      const SmallPtrSet<void *, 8> &GlobalLinkedBinaries,
      const std::string &DeviceArch, const CodeGenerationConfig &CGConfig,
      bool DumpIR, bool RelinkGlobalsByCopy)
      : Bitcode(Bitcode), HashValue(HashValue), KernelName(KernelName),
        Suffix(Suffix), BlockDim(BlockDim), GridDim(GridDim), RCVec(RCVec),
        LambdaCalleeInfo(LambdaCalleeInfo),
        VarNameToGlobalInfo(VarNameToGlobalInfo),
        GlobalLinkedBinaries(GlobalLinkedBinaries), DeviceArch(DeviceArch),
        CGOption(CGConfig.codeGenOption()), DumpIR(DumpIR),
        RelinkGlobalsByCopy(RelinkGlobalsByCopy),
        MinBlocksPerSM(
            CGConfig.minBlocksPerSM(BlockDim.x * BlockDim.y * BlockDim.z)),
        SpecializeArgs(CGConfig.specializeArgs()),
        SpecializeDims(CGConfig.specializeDims()),
        SpecializeDimsRange(CGConfig.specializeDimsRange()),
        SpecializeLaunchBounds(CGConfig.specializeLaunchBounds()),
        OptLevel(CGConfig.optLevel()),
        CodegenOptLevel(CGConfig.codeGenOptLevel()),
        PassPipeline(CGConfig.optPipeline()) {
    if (Config::get().ProteusTraceOutput >= 1) {
      llvm::SmallString<128> S;
      llvm::raw_svector_ostream OS(S);
      OS << "[KernelConfig] ID:" << KernelName << " ";
      CGConfig.dump(OS);
      OS << "\n";
      Logger::trace(OS.str());
    }
  }

  // Delete copy operations.
  CompilationTask(const CompilationTask &) = delete;
  CompilationTask &operator=(const CompilationTask &) = delete;

  // Use default move operations.
  CompilationTask(CompilationTask &&) noexcept = default;
  CompilationTask &operator=(CompilationTask &&) noexcept = default;

  HashT getHashValue() const { return HashValue; }

  std::unique_ptr<MemoryBuffer> compile() {
    struct TimerRAII {
      std::chrono::high_resolution_clock::time_point Start, End;
      HashT HashValue;
      TimerRAII(HashT HashValue) : HashValue(HashValue) {
        if (Config::get().ProteusDebugOutput) {
          Start = std::chrono::high_resolution_clock::now();
        }
      }

      ~TimerRAII() {
        if (Config::get().ProteusDebugOutput) {
          auto End = std::chrono::high_resolution_clock::now();
          auto Duration = End - Start;
          auto Milliseconds =
              std::chrono::duration_cast<std::chrono::milliseconds>(Duration)
                  .count();
          Logger::logs("proteus")
              << "Compiled HashValue " << HashValue.toString() << " for "
              << Milliseconds << "ms\n";
        }
      }
    } Timer{HashValue};

    LLVMContext Ctx;
    std::unique_ptr<Module> M = cloneKernelModule(Ctx);

    std::string KernelMangled = (KernelName + Suffix);

    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".input.ll", *M));

    proteus::specializeIR(*M, KernelName, Suffix, BlockDim, GridDim, RCVec,
                          LambdaCalleeInfo, SpecializeArgs, SpecializeDims,
                          SpecializeDimsRange, SpecializeLaunchBounds,
                          MinBlocksPerSM);

    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".specialized.ll", *M));

    replaceGlobalVariablesWithPointers(*M, VarNameToGlobalInfo);

    invokeOptimizeIR(*M);
    if (Config::get().ProteusTraceOutput == 2) {
      llvm::outs() << "LLVM IR module post optimization " << *M << "\n";
    }
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
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(),
                                   VarNameToGlobalInfo);

    return ObjBuf;
  }
};

} // namespace proteus

#endif
