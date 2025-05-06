#ifndef PROTEUS_CORE_LLVM_HIP_HPP
#define PROTEUS_CORE_LLVM_HIP_HPP

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#if LLVM_VERSION_MAJOR == 18
#include <lld/Common/Driver.h>
LLD_HAS_DRIVER(elf)
#endif

#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/UtilsHIP.h"

namespace proteus {

using namespace llvm;

namespace detail {

inline const SmallVector<StringRef> &gridDimXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv",
      "llvm.amdgcn.num.workgroups.x"};
  return Names;
}

inline const SmallVector<StringRef> &gridDimYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv",
      "llvm.amdgcn.num.workgroups.y"};
  return Names;
}

inline const SmallVector<StringRef> &gridDimZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv",
      "llvm.amdgcn.num.workgroups.z"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimXFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv",
      "llvm.amdgcn.workgroup.size.x"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimYFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv",
      "llvm.amdgcn.workgroup.size.y"};
  return Names;
}

inline const SmallVector<StringRef> &blockDimZFnName() {
  static SmallVector<StringRef> Names = {
      "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv",
      "llvm.amdgcn.workgroup.size.z"};
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

} // namespace detail

inline void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                     int BlockSize) {
  // TODO: fix calculation of launch bounds.
  // TODO: find maximum (hardcoded 1024) from device info.
  // TODO: Setting as 1, BlockSize to replicate launch bounds settings
  // Does setting it as BlockSize, BlockSize help?
  // Setting the attribute override any previous setting.
  F.addFnAttr("amdgpu-flat-work-group-size",
              "1," + std::to_string(std::min(1024, BlockSize)));
  // TODO: find warp size (hardcoded 64) from device info.
  // int WavesPerEU = (GridSize * BlockSize) / 64 / 110 / 4 / 2;
  [[maybe_unused]] int WavesPerEU = 0;
  // F->addFnAttr("amdgpu-waves-per-eu", std::to_string(WavesPerEU));
  PROTEUS_DBG(Logger::logs("proteus")
              << "BlockSize " << BlockSize << " GridSize " << GridSize
              << " => Set Wokgroup size " << BlockSize
              << " WavesPerEU (unused) " << WavesPerEU << "\n");
}

inline std::unique_ptr<MemoryBuffer>
codegenObject(Module &M, StringRef DeviceArch,
              SmallPtrSetImpl<void *> &GlobalLinkedBinaries,
              bool UseRTC = true) {
  assert(GlobalLinkedBinaries.empty() &&
         "Expected empty linked binaries for HIP");
  Timer T;
  if (UseRTC) {
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
        HipLinkStatePtr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
        (void *)ModuleBuf.data(), ModuleBuf.size(), "", 0, nullptr, nullptr));
    proteusHiprtcErrCheck(
        hiprtcLinkComplete(HipLinkStatePtr, (void **)&BinOut, &BinSize));

    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "HIP RTC codegen " << T.elapsed() << " ms\n");

    return MemoryBuffer::getMemBuffer(StringRef{BinOut, BinSize});
  }

#if LLVM_VERSION_MAJOR == 18
  auto TMExpected = proteus::detail::createTargetMachine(M, DeviceArch);
  if (!TMExpected)
    PROTEUS_FATAL_ERROR(toString(TMExpected.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  SmallVector<char, 4096> ObjectCode;
  raw_svector_ostream OS(ObjectCode);
  TM->addPassesToEmitFile(PM, OS, nullptr, CodeGenFileType::ObjectFile,
                          /* DisableVerify */ true, MMIWP);

  PM.run(M);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "Codegen object " << T.elapsed() << " ms\n");

  T.reset();
  SmallString<64> TempDir;
  SmallString<64> ObjectPath;
  SmallString<64> SharedObjectPath;
  // The LLD linker interfaces are not thread-safe, so we use a mutex.
  static std::mutex Mutex;
  {
    sys::path::system_temp_directory(true, TempDir);
    int ObjectFD;
    if (auto EC = sys::fs::createUniqueFile(TempDir + "/proteus-jit-%%%%%%%.o",
                                            ObjectFD, ObjectPath))
      PROTEUS_FATAL_ERROR(EC.message());

    raw_fd_ostream OS(ObjectFD, true);
    OS << StringRef{ObjectCode.data(), ObjectCode.size()};
    OS.close();

    if (auto EC = sys::fs::createUniqueFile(TempDir + "/proteus-jit-%%%%%%%.so",
                                            SharedObjectPath))
      PROTEUS_FATAL_ERROR(EC.message());

    std::vector<const char *> Args{"ld.lld",  "--no-undefined",
                                   "-shared", ObjectPath.c_str(),
                                   "-o",      SharedObjectPath.c_str()};

    {
      std::lock_guard LockGuard{Mutex};
      lld::Result S = lld::lldMain(Args, llvm::outs(), llvm::errs(),
                                   {{lld::Gnu, &lld::elf::link}});
      if (S.retCode)
        PROTEUS_FATAL_ERROR("Error: lld failed");
    }
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFileAsStream(SharedObjectPath);
  if (!Buffer)
    PROTEUS_FATAL_ERROR("Error reading file: " + Buffer.getError().message());

  sys::fs::remove(ObjectPath);
  sys::fs::remove(SharedObjectPath);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "Codegen linking " << T.elapsed() << " ms\n");

  return std::move(*Buffer);
#else
  PROTEUS_FATAL_ERROR("Expected LLVM18 for non-RTC codegen");
#endif
}

} // namespace proteus

#endif
