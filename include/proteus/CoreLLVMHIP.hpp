#ifndef PROTEUS_CORE_LLVM_HIP_HPP
#define PROTEUS_CORE_LLVM_HIP_HPP

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/LTO/LTO.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Target/TargetMachine.h>

#if LLVM_VERSION_MAJOR >= 18
#include <lld/Common/Driver.h>
LLD_HAS_DRIVER(elf)
#endif

#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/Utils.h"
#include "proteus/UtilsHIP.h"

namespace proteus {

using namespace llvm;

namespace detail {

inline const SmallVector<StringRef> &gridDimXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv",
      "llvm.amdgcn.num.workgroups.x", "_ZL20__hip_get_grid_dim_xv"};
  return Names;
}

inline const SmallVector<StringRef> &gridDimYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv",
      "llvm.amdgcn.num.workgroups.y", "_ZL20__hip_get_grid_dim_yv"};
  return Names;
}

inline const SmallVector<StringRef> &gridDimZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv",
      "llvm.amdgcn.num.workgroups.z", "_ZL20__hip_get_grid_dim_zv"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv",
      "llvm.amdgcn.workgroup.size.x", "_ZL21__hip_get_block_dim_xv"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv",
      "llvm.amdgcn.workgroup.size.y", "_ZL21__hip_get_block_dim_yv"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv",
      "llvm.amdgcn.workgroup.size.z", "_ZL21__hip_get_block_dim_zv"};
  return Names;
}

inline const SmallVector<StringRef> &blockIdxXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__XcvjEv",
      "llvm.amdgcn.workgroup.id.x"};
  return Names;
};

inline const SmallVector<StringRef> &blockIdxYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__YcvjEv",
      "llvm.amdgcn.workgroup.id.y"};
  return Names;
};

inline const SmallVector<StringRef> &blockIdxZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__ZcvjEv",
      "llvm.amdgcn.workgroup.id.z"};
  return Names;
}

inline const SmallVector<StringRef> &threadIdxXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__XcvjEv",
      "llvm.amdgcn.workitem.id.x"};
  return Names;
};

inline const SmallVector<StringRef> &threadIdxYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__YcvjEv",
      "llvm.amdgcn.workitem.id.y"};
  return Names;
};

inline const SmallVector<StringRef> &threadIdxZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__ZcvjEv",
      "llvm.amdgcn.workitem.id.z"};
  return Names;
};

inline Expected<sys::fs::TempFile> createTempFile(StringRef Prefix,
                                                  StringRef Suffix) {
  SmallString<128> TmpDir;
  sys::path::system_temp_directory(true, TmpDir);

  SmallString<64> FileName;
  FileName.append(Prefix);
  FileName.append(Suffix.empty() ? "-%%%%%%%" : "-%%%%%%%.");
  FileName.append(Suffix);
  sys::path::append(TmpDir, FileName);
  return sys::fs::TempFile::create(TmpDir);
}

#if LLVM_VERSION_MAJOR >= 18
inline SmallVector<std::unique_ptr<sys::fs::TempFile>>
codegenSerial(Module &M, StringRef DeviceArch,
              [[maybe_unused]] char OptLevel = '3', int CodegenOptLevel = 3) {
  SmallVector<std::unique_ptr<sys::fs::TempFile>> ObjectFiles;

  auto ExpectedTM =
      proteus::detail::createTargetMachine(M, DeviceArch, CodegenOptLevel);
  if (!ExpectedTM)
    PROTEUS_FATAL_ERROR(toString(ExpectedTM.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*ExpectedTM);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  M.setDataLayout(TM->createDataLayout());

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  SmallVector<char, 4096> ObjectCode;
  raw_svector_ostream OS(ObjectCode);
  auto ExpectedF = createTempFile("object", "o");
  if (auto E = ExpectedF.takeError())
    PROTEUS_FATAL_ERROR("Error creating object tmp file " +
                        toString(std::move(E)));
  auto ObjectFile = std::move(*ExpectedF);
  auto FileStream = std::make_unique<CachedFileStream>(
      std::make_unique<llvm::raw_fd_ostream>(ObjectFile.FD, false));
  TM->addPassesToEmitFile(PM, *FileStream->OS, nullptr,
                          CodeGenFileType::ObjectFile,
                          /* DisableVerify */ true, MMIWP);

  std::unique_ptr<sys::fs::TempFile> ObjectFilePtr =
      std::make_unique<sys::fs::TempFile>(std::move(ObjectFile));
  ObjectFiles.emplace_back(std::move(ObjectFilePtr));

  PM.run(M);

  return ObjectFiles;
}

inline SmallVector<std::unique_ptr<sys::fs::TempFile>>
codegenParallel(Module &M, StringRef DeviceArch, unsigned int OptLevel = 3,
                int CodegenOptLevel = 3) {
  // Use regular LTO with parallelism enabled to parallelize codegen.
  std::atomic<bool> LTOError = false;

  auto DiagnosticHandler = [&](const DiagnosticInfo &DI) {
    std::string ErrStorage;
    raw_string_ostream OS(ErrStorage);
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);

    switch (DI.getSeverity()) {
    case DS_Error:
      WithColor::error(errs(), "[proteus codegen]") << ErrStorage << "\n";
      LTOError = true;
      break;
    case DS_Warning:
      WithColor::warning(errs(), "[proteus codegen]") << ErrStorage << "\n";
      break;
    case DS_Note:
      WithColor::note(errs(), "[proteus codegen]") << ErrStorage << "\n";
      break;
    case DS_Remark:
      WithColor::remark(errs()) << ErrStorage << "\n";
      break;
    }
  };

  lto::Config Conf;
  Conf.CPU = DeviceArch;
  // Use default machine attributes.
  Conf.MAttrs = {};
  Conf.DisableVerify = true;
  Conf.TimeTraceEnabled = false;
  Conf.DebugPassManager = false;
  Conf.VerifyEach = false;
  Conf.DiagHandler = DiagnosticHandler;
  Conf.OptLevel = OptLevel;
  Conf.CGOptLevel = static_cast<CodeGenOptLevel>(CodegenOptLevel);

  unsigned ParallelCodeGenParallelismLevel =
      std::max(1u, std::thread::hardware_concurrency());
  lto::LTO L(std::move(Conf), nullptr, ParallelCodeGenParallelismLevel);

  SmallString<0> BitcodeBuf;
  raw_svector_ostream BitcodeOS(BitcodeBuf);
  WriteBitcodeToFile(M, BitcodeOS);

  // TODO: Module identifier can be empty because you always have on module to
  // link. However, in the general case, with multiple modules, each one must
  // have a unique identifier for LTO to work correctly.
  auto IF = cantFail(lto::InputFile::create(
      MemoryBufferRef{BitcodeBuf, M.getModuleIdentifier()}));

  std::set<std::string> PrevailingSymbols;
  auto BuildResolutions = [&]() {
    // Save the input file and the buffer associated with its memory.
    const auto Symbols = IF->symbols();
    SmallVector<lto::SymbolResolution, 16> Resolutions(Symbols.size());
    size_t SymbolIdx = 0;
    for (auto &Sym : Symbols) {
      lto::SymbolResolution &Res = Resolutions[SymbolIdx];
      SymbolIdx++;

      // All defined symbols are prevailing.
      Res.Prevailing = !Sym.isUndefined() &&
                       PrevailingSymbols.insert(Sym.getName().str()).second;

      Res.VisibleToRegularObj =
          Res.Prevailing &&
          Sym.getVisibility() != GlobalValue::HiddenVisibility &&
          !Sym.canBeOmittedFromSymbolTable();

      Res.ExportDynamic =
          Sym.getVisibility() != GlobalValue::HiddenVisibility &&
          (!Sym.canBeOmittedFromSymbolTable());

      Res.FinalDefinitionInLinkageUnit =
          Sym.getVisibility() != GlobalValue::DefaultVisibility &&
          (!Sym.isUndefined() && !Sym.isCommon());

      // Device linking does not support linker redefined symbols (e.g. --wrap).
      Res.LinkerRedefined = false;

#if PROTEUS_ENABLE_DEBUG
      auto PrintSymbol = [](const lto::InputFile::Symbol &Sym,
                            lto::SymbolResolution &Res) {
        auto &OutStream = Logger::logs("proteus");
        OutStream << "Vis: ";
        switch (Sym.getVisibility()) {
        case GlobalValue::HiddenVisibility:
          OutStream << 'H';
          break;
        case GlobalValue::ProtectedVisibility:
          OutStream << 'P';
          break;
        case GlobalValue::DefaultVisibility:
          OutStream << 'D';
          break;
        }

        OutStream << " Sym: ";
        auto PrintBool = [&](char C, bool B) { OutStream << (B ? C : '-'); };
        PrintBool('U', Sym.isUndefined());
        PrintBool('C', Sym.isCommon());
        PrintBool('W', Sym.isWeak());
        PrintBool('I', Sym.isIndirect());
        PrintBool('O', Sym.canBeOmittedFromSymbolTable());
        PrintBool('T', Sym.isTLS());
        PrintBool('X', Sym.isExecutable());
        OutStream << ' ' << Sym.getName();
        OutStream << "| P " << Res.Prevailing;
        OutStream << " V " << Res.VisibleToRegularObj;
        OutStream << " E " << Res.ExportDynamic;
        OutStream << " F " << Res.FinalDefinitionInLinkageUnit;
        OutStream << "\n";
      };
      PrintSymbol(Sym, Res);
#endif
    }

    // Add the bitcode file with its resolved symbols to the LTO job.
    cantFail(L.add(std::move(IF), Resolutions));
  };

  BuildResolutions();

  // Run the LTO job to compile the bitcode.
  size_t MaxTasks = L.getMaxTasks();
  SmallVector<std::unique_ptr<sys::fs::TempFile>> ObjectFiles{MaxTasks};

  auto AddStream =
      [&](size_t Task,
          const Twine & /*ModuleName*/) -> std::unique_ptr<CachedFileStream> {
    std::string TaskStr = Task ? "." + std::to_string(Task) : "";
    auto ExpectedF = createTempFile("lto-shard" + TaskStr, "o");
    if (auto E = ExpectedF.takeError())
      PROTEUS_FATAL_ERROR("Error creating tmp file " + toString(std::move(E)));
    ObjectFiles[Task] =
        std::make_unique<sys::fs::TempFile>(std::move(*ExpectedF));
    auto Ret = std::make_unique<CachedFileStream>(
        std::make_unique<llvm::raw_fd_ostream>(ObjectFiles[Task]->FD, false));
    if (!Ret)
      PROTEUS_FATAL_ERROR("Error creating CachedFileStream");
    return Ret;
  };

  if (Error E = L.run(AddStream))
    PROTEUS_FATAL_ERROR("Error: " + toString(std::move(E)));

  if (LTOError)
    PROTEUS_FATAL_ERROR(toString(
        createStringError(inconvertibleErrorCode(),
                          "Errors encountered inside the LTO pipeline.")));

  return ObjectFiles;
}
#endif

inline std::unique_ptr<MemoryBuffer> codegenRTC(Module &M,
                                                StringRef DeviceArch) {
  char *BinOut;
  size_t BinSize;

  SmallString<4096> ModuleBuf;
  raw_svector_ostream ModuleBufOS(ModuleBuf);
  WriteBitcodeToFile(M, ModuleBufOS);

  hiprtcLinkState HipLinkStatePtr;

  // NOTE: This code is an example of passing custom, AMD-specific
  // options to the compiler/linker.
  // NOTE: Unrolling can have a dramatic (time-consuming) effect on JIT
  // compilation time and on the resulting optimization, better or worse
  // depending on code specifics.
  std::string MArchOpt = ("-march=" + DeviceArch).str();
  const char *OptArgs[] = {"-mllvm", "-unroll-threshold=1000",
                           MArchOpt.c_str()};
  std::vector<hiprtcJIT_option> JITOptions = {
      HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t OptArgsSize = 3;
  const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
  proteusHiprtcErrCheck(hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                                         (void **)JITOptionsValues,
                                         &HipLinkStatePtr));
  // NOTE: the following version of te code does not set options.
  // proteusHiprtcErrCheck(hiprtcLinkCreate(0, nullptr, nullptr,
  // &hip_link_state_ptr));

  proteusHiprtcErrCheck(hiprtcLinkAddData(
      HipLinkStatePtr, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void *)ModuleBuf.data(),
      ModuleBuf.size(), "", 0, nullptr, nullptr));
  proteusHiprtcErrCheck(
      hiprtcLinkComplete(HipLinkStatePtr, (void **)&BinOut, &BinSize));

  return MemoryBuffer::getMemBuffer(StringRef{BinOut, BinSize});
}

} // namespace detail

inline void setLaunchBoundsForKernel(Function &F, int MaxNumWorkGroups,
                                     int WavesPerEU = 0) {
  // TODO: fix calculation of launch bounds.
  // TODO: find maximum (hardcoded 1024) from device info.
  // TODO: Setting as 1, BlockSize to replicate launch bounds settings
  F.addFnAttr("amdgpu-flat-work-group-size",
              "1," + std::to_string(std::min(1024, MaxNumWorkGroups)));
  // F->addFnAttr("amdgpu-waves-per-eu", std::to_string(WavesPerEU));
  if (WavesPerEU != 0) {
    // NOTE: We are missing a heuristic to define the `WavesPerEU`, as such we
    // still need to study it. I restrict the waves by setting min equal to max
    // and disallowing any heuristics that HIP will use internally.
    // For more information please check:
    // https://clang.llvm.org/docs/AttributeReference.html#amdgpu-waves-per-eu
    F.addFnAttr("amdgpu-waves-per-eu",
                std::to_string(WavesPerEU) + "," + std::to_string(WavesPerEU));
  }

  PROTEUS_DBG(Logger::logs("proteus")
              << " => Set Workgroup size " << MaxNumWorkGroups
              << " WavesPerEU (unused) " << WavesPerEU << "\n");
}

inline std::unique_ptr<MemoryBuffer>
codegenObject(Module &M, StringRef DeviceArch,
              [[maybe_unused]] SmallPtrSetImpl<void *> &GlobalLinkedBinaries,
              CodegenOption CGOption = CodegenOption::RTC) {
  assert(GlobalLinkedBinaries.empty() &&
         "Expected empty linked binaries for HIP");
  Timer T;
  SmallVector<std::unique_ptr<sys::fs::TempFile>> ObjectFiles;
  switch (CGOption) {
  case CodegenOption::RTC: {
    auto Ret = detail::codegenRTC(M, DeviceArch);
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Codegen RTC " << T.elapsed() << " ms\n");
    return Ret;
  }
#if LLVM_VERSION_MAJOR >= 18
  case CodegenOption::Serial:
    ObjectFiles = detail::codegenSerial(M, DeviceArch);
    break;
  case CodegenOption::Parallel:
    ObjectFiles = detail::codegenParallel(M, DeviceArch);
    break;
#endif
  default:
    PROTEUS_FATAL_ERROR("Unknown Codegen Option");
  }

  if (ObjectFiles.empty())
    PROTEUS_FATAL_ERROR("Expected non-empty vector of object files");

#if LLVM_VERSION_MAJOR >= 18
  auto ExpectedF = detail::createTempFile("proteus-jit", "o");
  if (auto E = ExpectedF.takeError())
    PROTEUS_FATAL_ERROR("Error creating shared object file " +
                        toString(std::move(E)));

  auto SharedObject = std::move(*ExpectedF);

  std::vector<const char *> Args{"ld.lld", "--no-undefined", "-shared", "-o",
                                 SharedObject.TmpName.c_str()};
  for (auto &File : ObjectFiles) {
    if (!File)
      continue;
    Args.push_back(File->TmpName.c_str());
  }

#if PROTEUS_ENABLE_DEBUG
  for (auto &Arg : Args) {
    Logger::logs("proteus") << Arg << " ";
  }
  Logger::logs("proteus") << "\n";
#endif

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "Codegen object " << toString(CGOption) << "["
                       << ObjectFiles.size() << "] " << T.elapsed() << " ms\n");

  T.reset();
  // The LLD linker interface is not thread-safe, so we use a mutex.
  static std::mutex Mutex;
  {
    std::lock_guard LockGuard{Mutex};
    lld::Result S = lld::lldMain(Args, llvm::outs(), llvm::errs(),
                                 {{lld::Gnu, &lld::elf::link}});
    if (S.retCode)
      PROTEUS_FATAL_ERROR("Error: lld failed");
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFileAsStream(SharedObject.TmpName);
  if (!Buffer)
    PROTEUS_FATAL_ERROR("Error reading file: " + Buffer.getError().message());

  // Remove temporary files.
  for (auto &File : ObjectFiles) {
    if (!File)
      continue;
    if (auto E = File->discard())
      PROTEUS_FATAL_ERROR("Error removing object tmp file " +
                          toString(std::move(E)));
  }
  if (auto E = SharedObject.discard())
    PROTEUS_FATAL_ERROR("Error removing shared object tmp file " +
                        toString(std::move(E)));

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "Codegen linking " << T.elapsed() << " ms\n");

  return std::move(*Buffer);
#else
  PROTEUS_FATAL_ERROR("Expected LLVM18 for non-RTC codegen");
#endif
}

} // namespace proteus

#endif
