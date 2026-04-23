#ifndef PROTEUS_COMPILATION_TASK_H
#define PROTEUS_COMPILATION_TASK_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/CoreLLVMDevice.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/Utils.h"

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
  OptimizationPipelineConfig OptConfig;

  std::unique_ptr<Module> cloneKernelModule(LLVMContext &Ctx) {
    TIMESCOPE(CompilationTask, cloneKernelModule);
    auto ClonedModule = parseBitcodeFile(Bitcode, Ctx);
    if (auto E = ClonedModule.takeError()) {
      reportFatalError("Failed to parse bitcode" + toString(std::move(E)));
    }

    return std::move(*ClonedModule);
  }

  void invokeOptimizeIR(Module &M) {
    TIMESCOPE(CompilationTask, invokeOptimizeIR);
#if PROTEUS_ENABLE_CUDA
    // For CUDA we always run the optimization pipeline.
    optimizeIR(M, DeviceArch, OptConfig);
#elif PROTEUS_ENABLE_HIP
    // For HIP we run the optimization pipeline here only for Serial codegen.
    // Parallel codegen forwards custom pipelines to LTO; HIP RTC invokes
    // optimization internally.
    // TODO: Move optimizeIR inside the codegen routines?
    if (CGOption == CodegenOption::Serial)
      optimizeIR(M, DeviceArch, OptConfig);
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
        OptConfig(CGConfig) {
    if (Config::get().traceSpecializations()) {
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
    TIMESCOPE(CompilationTask, compile);
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
    if (Config::get().traceIRDump()) {
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

#if PROTEUS_ENABLE_CUDA
    auto ObjBuf =
        proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries, CGOption);
#elif PROTEUS_ENABLE_HIP
    auto ObjBuf = proteus::codegenObject(*M, DeviceArch, GlobalLinkedBinaries,
                                         CGOption, OptConfig);
#endif

    if (!RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(),
                                   VarNameToGlobalInfo);

    return ObjBuf;
  }
};

} // namespace proteus

#endif
