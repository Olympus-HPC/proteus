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
#include <memory>
#include <string>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/ELF.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "proteus/CoreLLVM.hpp"
#include "proteus/CoreLLVMHIP.hpp"
#include "proteus/JitEngineDeviceHIP.hpp"
#include "proteus/TimeTracing.hpp"

#if LLVM_VERSION_MAJOR == 18
#include <lld/Common/Driver.h>
LLD_HAS_DRIVER(elf)
#endif

using namespace proteus;
using namespace llvm;

void *JitEngineDeviceHIP::resolveDeviceGlobalAddr(const void *Addr) {
  return proteus::resolveDeviceGlobalAddr(Addr);
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
    PROTEUS_FATAL_ERROR("Error missing magic string");
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
    PROTEUS_FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    PROTEUS_FATAL_ERROR("Error reading sections");

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
      PROTEUS_FATAL_ERROR("Error reading section contents");
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
      PROTEUS_FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto SectionHashStr =
        SectionName->slice(SectionName->find_last_of(".") + 1, StringRef::npos);
    HashT SectionHashValue{SectionHashStr};

    // NOTE: We include the hash value of the LTO section, which encodes changes
    // to non-proteus compiled external modules.
    BinInfo.updateModuleHash(SectionHashValue);
  }

  return BinInfo.getModuleHash();
}

std::unique_ptr<Module> JitEngineDeviceHIP::extractModule(BinaryInfo &BinInfo) {
  Expected<object::ELF64LEFile> DeviceElf =
      object::ELF64LEFile::create(getDeviceBinary(BinInfo, DeviceArch));
  if (DeviceElf.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    PROTEUS_FATAL_ERROR("Error reading sections");

  ArrayRef<uint8_t> DeviceBitcode;
  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = getLLVMContext();

  auto ExtractModuleFromSection = [&DeviceElf, &Ctx](auto &Section,
                                                     StringRef SectionName) {
    ArrayRef<uint8_t> BitcodeData;
    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      PROTEUS_FATAL_ERROR("Error reading section contents");
    BitcodeData = *SectionContents;
    auto Bitcode = StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                             BitcodeData.size()};

    SMDiagnostic Err;
    auto M = parseIR(MemoryBufferRef{Bitcode, SectionName}, Err, Ctx);
    if (!M)
      PROTEUS_FATAL_ERROR("unexpected");

    return M;
  };

  // We extract bitcode from sections. If there is a .jit.bitcode.lto section
  // due to RDC compilation, we keep it separately to import definitions as
  // needed at linking.
  std::unique_ptr<Module> LTOModule = nullptr;
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      PROTEUS_FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto M = ExtractModuleFromSection(Section, *SectionName);

    if (SectionName->starts_with(".jit.bitcode.lto")) {
      if (LTOModule)
        PROTEUS_FATAL_ERROR("Expected single LTO Module");
      LTOModule = std::move(M);
      continue;
    }

    LinkedModules.push_back(std::move(M));
  }

  return linkJitModule(LinkedModules, std::move(LTOModule));
}

void JitEngineDeviceHIP::setLaunchBoundsForKernel(Module &M, Function &F,
                                                  size_t GridSize,
                                                  int BlockSize) {
  proteus::setLaunchBoundsForKernel(M, F, GridSize, BlockSize);
}

std::unique_ptr<MemoryBuffer>
JitEngineDeviceHIP::codegenObject(Module &M, StringRef DeviceArch) {
  TIMESCOPE("Codegen object");
  return proteus::codegenObject(M, DeviceArch, GlobalLinkedBinaries,
                                Config.ENV_PROTEUS_USE_HIP_RTC_CODEGEN);
}

hipFunction_t
JitEngineDeviceHIP::getKernelFunctionFromImage(StringRef KernelName,
                                               const void *Image) {
  return proteus::getKernelFunctionFromImage(
      KernelName, Image, Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY,
      VarNameToDevPtr);
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

JitEngineDeviceHIP::JitEngineDeviceHIP() {
  proteus::InitAMDGPUTarget();
  hipDeviceProp_t DevProp;
  proteusHipErrCheck(hipGetDeviceProperties(&DevProp, 0));

  DeviceArch = DevProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));
}
