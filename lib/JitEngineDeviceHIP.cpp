//===-- JitEngineDeviceHIP.cpp -- JIT Engine Device for HIP impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <memory>
#include <string>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "JitEngineDevice.hpp"
#include "JitEngineDeviceHIP.hpp"
#include "TimeTracing.hpp"
#include "Utils.h"

#if LLVM_VERSION_MAJOR == 18
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif

using namespace proteus;
using namespace llvm;

void *JitEngineDeviceHIP::resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  proteusHipErrCheck(hipGetSymbolAddress(&DevPtr, HIP_SYMBOL(Addr)));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
}

JitEngineDeviceHIP &JitEngineDeviceHIP::instance() {
  static JitEngineDeviceHIP Jit{};
  return Jit;
}

static StringRef getDeviceBinary(BinaryInfo &BinInfo, StringRef DeviceArch) {
  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();

  constexpr char OffloadBundlerMagicStr[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t Pos = 0;

  const char *Binary = FatbinWrapper->Binary;

  StringRef Magic(Binary, sizeof(OffloadBundlerMagicStr) - 1);
  if (!Magic.equals(OffloadBundlerMagicStr))
    FATAL_ERROR("Error missing magic string");
  Pos += sizeof(OffloadBundlerMagicStr) - 1;

  auto Read8ByteIntLE = [](const char *S, size_t Pos) {
    return support::endian::read64le(S + Pos);
  };

  uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
  Pos += 8;
  PROTEUS_DBG(Logger::logs("proteus")
              << "NumberOfbundles " << NumberOfBundles << "\n");

  StringRef DeviceBinary;
  for (uint64_t I = 0; I < NumberOfBundles; ++I) {
    uint64_t Offset = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    uint64_t Size = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    uint64_t TripleSize = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    StringRef Triple(Binary + Pos, TripleSize);
    Pos += TripleSize;

    PROTEUS_DBG(Logger::logs("proteus") << "Offset " << Offset << "\n");
    PROTEUS_DBG(Logger::logs("proteus") << "Size " << Size << "\n");
    PROTEUS_DBG(Logger::logs("proteus") << "TripleSize " << TripleSize << "\n");
    PROTEUS_DBG(Logger::logs("proteus") << "Triple " << Triple << "\n");

    if (!Triple.contains("amdgcn") || !Triple.contains(DeviceArch)) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "mismatching architecture, skipping ...\n");
      continue;
    }

    DeviceBinary = StringRef{Binary + Offset, Size};
    break;
  }

  return DeviceBinary;
}

HashT JitEngineDeviceHIP::getModuleHash(BinaryInfo &BinInfo) {
  if (BinInfo.hasModuleHash())
    return BinInfo.getModuleHash();

  Expected<object::ELF64LEFile> DeviceElf =
      object::ELF64LEFile::create(getDeviceBinary(BinInfo, DeviceArch));
  if (DeviceElf.takeError())
    FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    FATAL_ERROR("Error reading sections");

  ArrayRef<uint8_t> DeviceBitcode;

  // NOTE: This code hashes the bitcode of the section. Leaving it here in case
  // there is a reason to revert to computing the hash at runtime instead of
  // compilation time.
  /*
  auto HashSectionBitcode = [&DeviceElf](auto &Section, StringRef
   SectionName) {
    ArrayRef<uint8_t> BitcodeData;
    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      FATAL_ERROR("Error reading section contents");
    BitcodeData = *SectionContents;
    auto Bitcode = StringRef{reinterpret_cast<const char
    *>(BitcodeData.data()),
                             BitcodeData.size()};
    return hash(Bitcode);
  };
  */

  // We comibine hash values from sections storing bitcodes. If there is a
  // .jit.bitcode.lto section due to RDC compilation that's the only hash we
  // need.
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto SectionHashStr =
        SectionName->slice(SectionName->find_last_of(".") + 1, StringRef::npos);
    HashT SectionHashValue{SectionHashStr};

    if (SectionName->starts_with(".jit.bitcode.lto")) {
      BinInfo.setModuleHash(SectionHashValue);
      break;
    }

    BinInfo.updateModuleHash(SectionHashValue);
  }

  return BinInfo.getModuleHash();
}

std::unique_ptr<Module> JitEngineDeviceHIP::extractModule(BinaryInfo &BinInfo) {
  Expected<object::ELF64LEFile> DeviceElf =
      object::ELF64LEFile::create(getDeviceBinary(BinInfo, DeviceArch));
  if (DeviceElf.takeError())
    FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    FATAL_ERROR("Error reading sections");

  ArrayRef<uint8_t> DeviceBitcode;
  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = getLLVMContext();

  auto ExtractModuleFromSection = [&DeviceElf, &Ctx](auto &Section,
                                                     StringRef SectionName) {
    ArrayRef<uint8_t> BitcodeData;
    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      FATAL_ERROR("Error reading section contents");
    BitcodeData = *SectionContents;
    auto Bitcode = StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                             BitcodeData.size()};

    SMDiagnostic Err;
    auto M = parseIR(MemoryBufferRef{Bitcode, SectionName}, Err, Ctx);
    if (!M)
      FATAL_ERROR("unexpected");

    return M;
  };

  // We extract bitcode from sections. If there is a .jit.bitcode.lto section
  // due to RDC compilation that's the only bitcode we need, othewise we
  // collect all .jit.bitcode sections.
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto M = ExtractModuleFromSection(Section, *SectionName);

    if (SectionName->starts_with(".jit.bitcode.lto")) {
      LinkedModules.clear();
      LinkedModules.push_back(std::move(M));
      break;
    }

    LinkedModules.push_back(std::move(M));
  }

  return linkJitModule(LinkedModules);
}

void JitEngineDeviceHIP::setLaunchBoundsForKernel(Module &M, Function &F,
                                                  size_t GridSize,
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
  int WavesPerEU = 0;
  // F->addFnAttr("amdgpu-waves-per-eu", std::to_string(WavesPerEU));
  PROTEUS_DBG(Logger::logs("proteus")
              << "BlockSize " << BlockSize << " GridSize " << GridSize
              << " => Set Wokgroup size " << BlockSize
              << " WavesPerEU (unused) " << WavesPerEU << "\n");
}

std::unique_ptr<MemoryBuffer>
JitEngineDeviceHIP::codegenObject(Module &M, StringRef DeviceArch) {
  TIMESCOPE("Codegen object");
#if LLVM_VERSION_MAJOR == 18
  if (Config.ENV_PROTEUS_USE_HIP_RTC_CODEGEN) {
#else
  {
#endif
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
    {
      TIMESCOPE("HIP_RTC")
      std::string MArchOpt = ("-march=" + DeviceArch).str();
      const char *OptArgs[] = {"-mllvm", "-unroll-threshold=1000",
                               MArchOpt.c_str()};
      std::vector<hiprtcJIT_option> JITOptions = {
          HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
      size_t OptArgsSize = 3;
      const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
      proteusHiprtcErrCheck(
          hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                           (void **)JITOptionsValues, &HipLinkStatePtr));
      // NOTE: the following version of te code does not set options.
      // proteusHiprtcErrCheck(hiprtcLinkCreate(0, nullptr, nullptr,
      // &hip_link_state_ptr));

      proteusHiprtcErrCheck(hiprtcLinkAddData(
          HipLinkStatePtr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
          (void *)ModuleBuf.data(), ModuleBuf.size(), "", 0, nullptr, nullptr));
      proteusHiprtcErrCheck(
          hiprtcLinkComplete(HipLinkStatePtr, (void **)&BinOut, &BinSize));
    }

    return MemoryBuffer::getMemBuffer(StringRef{BinOut, BinSize});
  }

#if LLVM_VERSION_MAJOR == 18
  auto TMExpected = createTargetMachine(M, DeviceArch);
  if (!TMExpected)
    FATAL_ERROR(toString(TMExpected.takeError()));

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

  SmallString<64> TempDir;
  SmallString<64> ObjectPath;
  SmallString<64> SharedObjectPath;
  {
    TIMESCOPE("LLD")
    sys::path::system_temp_directory(true, TempDir);
    int ObjectFD;
    if (auto EC = sys::fs::createUniqueFile(TempDir + "/proteus-jit-%%%%%%%.o",
                                            ObjectFD, ObjectPath))
      FATAL_ERROR(EC.message());

    raw_fd_ostream OS(ObjectFD, true);
    OS << StringRef{ObjectCode.data(), ObjectCode.size()};
    OS.close();

    if (auto EC = sys::fs::createUniqueFile(TempDir + "/proteus-jit-%%%%%%%.so",
                                            SharedObjectPath))
      FATAL_ERROR(EC.message());

    std::vector<const char *> Args{"ld.lld", "-shared", ObjectPath.c_str(),
                                   "-o", SharedObjectPath.c_str()};

    lld::Result S = lld::lldMain(Args, llvm::outs(), llvm::errs(),
                                 {{lld::Gnu, &lld::elf::link}});
    if (S.retCode)
      FATAL_ERROR("Error: lld failed");
  }

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(SharedObjectPath);
  if (!Buffer)
    FATAL_ERROR("Error reading file: " + Buffer.getError().message());

  sys::fs::remove(ObjectPath);
  sys::fs::remove(SharedObjectPath);

  return std::move(*Buffer);
#endif
}

hipFunction_t
JitEngineDeviceHIP::getKernelFunctionFromImage(StringRef KernelName,
                                               const void *Image) {
  hipModule_t HipModule;
  hipFunction_t KernelFunc;

  proteusHipErrCheck(hipModuleLoadData(&HipModule, Image));
  if (Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY) {
    for (auto &[GlobalName, HostAddr] : VarNameToDevPtr) {
      hipDeviceptr_t Dptr;
      size_t Bytes;
      proteusHipErrCheck(hipModuleGetGlobal(&Dptr, &Bytes, HipModule,
                                            (GlobalName + "$ptr").c_str()));

      void *DevPtr = resolveDeviceGlobalAddr(HostAddr);
      uint64_t PtrVal = (uint64_t)DevPtr;
      proteusHipErrCheck(hipMemcpyHtoD(Dptr, &PtrVal, Bytes));
    }
  }
  proteusHipErrCheck(
      hipModuleGetFunction(&KernelFunc, HipModule, KernelName.str().c_str()));

  return KernelFunc;
}

hipError_t JitEngineDeviceHIP::launchKernelFunction(hipFunction_t KernelFunc,
                                                    dim3 GridDim, dim3 BlockDim,
                                                    void **KernelArgs,
                                                    uint64_t ShmemSize,
                                                    hipStream_t Stream) {
  return hipModuleLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                               BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                               Stream, KernelArgs, nullptr);
}

hipError_t JitEngineDeviceHIP::launchKernelDirect(void *KernelFunc,
                                                  dim3 GridDim, dim3 BlockDim,
                                                  void **KernelArgs,
                                                  uint64_t ShmemSize,
                                                  hipStream_t Stream) {
  return hipLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

JitEngineDeviceHIP::JitEngineDeviceHIP() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  hipDeviceProp_t devProp;
  proteusHipErrCheck(hipGetDeviceProperties(&devProp, 0));

  DeviceArch = devProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));
}
