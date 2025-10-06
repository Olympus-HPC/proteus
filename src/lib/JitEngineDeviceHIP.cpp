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

#include "proteus/CoreDevice.hpp"
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

#if LLVM_VERSION_MAJOR >= 18
static Expected<std::unique_ptr<MemoryBuffer>> decompress(StringRef Blob) {
  static constexpr size_t MagicSize = 4;
  static constexpr size_t VersionFieldSize = sizeof(uint16_t);
  static constexpr size_t MethodFieldSize = sizeof(uint16_t);
  static constexpr size_t FileSizeFieldSize = sizeof(uint32_t);
  static constexpr size_t UncompressedSizeFieldSize = sizeof(uint32_t);
  static constexpr size_t HashFieldSize = sizeof(uint64_t);
  static constexpr size_t V2HeaderSize =
      MagicSize + VersionFieldSize + FileSizeFieldSize + MethodFieldSize +
      UncompressedSizeFieldSize + HashFieldSize;
  [[maybe_unused]] static constexpr llvm::StringRef MagicNumber = "CCOB";
  [[maybe_unused]] static constexpr uint16_t Version = 2;

  size_t CurrentOffset = MagicSize;

  uint16_t ThisVersion;
  memcpy(&ThisVersion, Blob.data() + CurrentOffset, sizeof(uint16_t));
  CurrentOffset += VersionFieldSize;

  uint16_t CompressionMethod;
  memcpy(&CompressionMethod, Blob.data() + CurrentOffset, sizeof(uint16_t));
  CurrentOffset += MethodFieldSize;

  uint32_t TotalFileSize;
  if (ThisVersion < 2)
    PROTEUS_FATAL_ERROR("V1 compressed offload bundle format is unsupported");

  memcpy(&TotalFileSize, Blob.data() + CurrentOffset, sizeof(uint32_t));
  CurrentOffset += FileSizeFieldSize;

  uint32_t UncompressedSize;
  memcpy(&UncompressedSize, Blob.data() + CurrentOffset, sizeof(uint32_t));
  CurrentOffset += UncompressedSizeFieldSize;

  uint64_t StoredHash;
  memcpy(&StoredHash, Blob.data() + CurrentOffset, sizeof(uint64_t));
  CurrentOffset += HashFieldSize;

  llvm::compression::Format CompressionFormat;
  if (CompressionMethod ==
      static_cast<uint16_t>(llvm::compression::Format::Zlib))
    CompressionFormat = llvm::compression::Format::Zlib;
  else if (CompressionMethod ==
           static_cast<uint16_t>(llvm::compression::Format::Zstd))
    CompressionFormat = llvm::compression::Format::Zstd;
  else
    return createStringError(inconvertibleErrorCode(),
                             "Unknown compressing method");

  SmallVector<uint8_t, 0> DecompressedData;
  StringRef CompressedData{Blob.data() + CurrentOffset,
                           TotalFileSize -
                               V2HeaderSize}; // = Blob.substr(CurrentOffset);
  if (llvm::Error DecompressionError = llvm::compression::decompress(
          CompressionFormat, llvm::arrayRefFromStringRef(CompressedData),
          DecompressedData, UncompressedSize))
    return createStringError(inconvertibleErrorCode(),
                             "Could not decompress embedded file contents: " +
                                 llvm::toString(std::move(DecompressionError)));

  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::toStringRef(DecompressedData));
}
#endif

static std::unique_ptr<MemoryBuffer> getDeviceBinary(BinaryInfo &BinInfo,
                                                     StringRef DeviceArch) {
  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();

#if LLVM_VERSION_MAJOR >= 18
  std::unique_ptr<MemoryBuffer> FatbinBuffer = nullptr;
  auto FileMagic = identify_magic(FatbinWrapper->Binary);
  switch (FileMagic) {
  case file_magic::offload_bundle:
    FatbinBuffer = MemoryBuffer::getMemBuffer(FatbinWrapper->Binary, "", false);
    break;
  case file_magic::offload_bundle_compressed: {
    auto ExpectedDecompressed = decompress(StringRef{FatbinWrapper->Binary});
    if (auto E = ExpectedDecompressed.takeError())
      PROTEUS_FATAL_ERROR("Error decompressing compressed offload bundle: " +
                          toString(std::move(E)));
    FatbinBuffer = std::move(*ExpectedDecompressed);
    break;
  }
  default:
    PROTEUS_FATAL_ERROR("Failed to read device binary bundle");
  }
#else
  std::unique_ptr<MemoryBuffer> FatbinBuffer =
      MemoryBuffer::getMemBuffer(FatbinWrapper->Binary, "", false);
#endif

  constexpr char OffloadBundlerMagicStr[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t Pos = 0;

  StringRef Magic(FatbinBuffer->getBufferStart(),
                  sizeof(OffloadBundlerMagicStr) - 1);
  if (!(Magic == OffloadBundlerMagicStr))
    PROTEUS_FATAL_ERROR("Error missing magic string");
  Pos += sizeof(OffloadBundlerMagicStr) - 1;

  auto Read8ByteIntLE = [](const char *S, size_t Pos) {
    return support::endian::read64le(S + Pos);
  };

  uint64_t NumberOfBundles =
      Read8ByteIntLE(FatbinBuffer->getBufferStart(), Pos);
  Pos += 8;
  PROTEUS_DBG(Logger::logs("proteus")
              << "NumberOfbundles " << NumberOfBundles << "\n");

  std::unique_ptr<MemoryBuffer> DeviceBinary;
  for (uint64_t I = 0; I < NumberOfBundles; ++I) {
    uint64_t Offset = Read8ByteIntLE(FatbinBuffer->getBufferStart(), Pos);
    Pos += 8;

    uint64_t Size = Read8ByteIntLE(FatbinBuffer->getBufferStart(), Pos);
    Pos += 8;

    uint64_t TripleSize = Read8ByteIntLE(FatbinBuffer->getBufferStart(), Pos);
    Pos += 8;

    StringRef Triple(FatbinBuffer->getBufferStart() + Pos, TripleSize);
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

    DeviceBinary = MemoryBuffer::getMemBuffer(
        StringRef{FatbinBuffer->getBufferStart() + Offset, Size}, "", false);
    break;
  }

  if (!DeviceBinary)
    PROTEUS_FATAL_ERROR("Expected non-null device binary");

  return MemoryBuffer::getMemBufferCopy(DeviceBinary->getBuffer());
}

static Expected<object::ELF64LEFile> getDeviceElf(BinaryInfo &BinInfo,
                                                  StringRef DeviceArch) {
  if (!BinInfo.hasDeviceBinary())
    BinInfo.setDeviceBinary(getDeviceBinary(BinInfo, DeviceArch));

  Expected<object::ELF64LEFile> DeviceElf =
      object::ELF64LEFile::create(BinInfo.getDeviceBinary().getBuffer());

  return DeviceElf;
}

HashT JitEngineDeviceHIP::getModuleHash(BinaryInfo &BinInfo) {
  if (BinInfo.hasModuleHash())
    return BinInfo.getModuleHash();

  Expected<object::ELF64LEFile> DeviceElf = getDeviceElf(BinInfo, DeviceArch);
  if (DeviceElf.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    PROTEUS_FATAL_ERROR("Error reading sections");

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

std::unique_ptr<Module> JitEngineDeviceHIP::tryExtractKernelModule(
    BinaryInfo &BinInfo, StringRef KernelName, LLVMContext &Ctx) {
  Expected<object::ELF64LEFile> DeviceElf = getDeviceElf(BinInfo, DeviceArch);
  if (DeviceElf.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    PROTEUS_FATAL_ERROR("Error reading sections");

  auto ExtractModuleFromSection = [&Ctx, &DeviceElf](auto &Section,
                                                     StringRef SectionName) {
    ArrayRef<uint8_t> BitcodeData;
    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      PROTEUS_FATAL_ERROR("Error reading section contents");
    BitcodeData = *SectionContents;
    auto Bitcode = StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                             BitcodeData.size()};

    Timer T;
    SMDiagnostic Diag;
    // Parse the IR module eagerly as it will be immediately used for codegen.
    auto M = parseIR(MemoryBufferRef(Bitcode, SectionName), Diag, Ctx);
    if (!M)
      PROTEUS_FATAL_ERROR("Error parsing IR: " + Diag.getMessage());
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Parse IR " << SectionName << " " << T.elapsed()
                         << " ms\n");

    return M;
  };

  // We extract the bitcode from the ELF sections uniquely identified by the
  // kernel symbol.
  std::unique_ptr<Module> KernelModule = nullptr;
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      PROTEUS_FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode." + KernelName.str()))
      continue;

    KernelModule = ExtractModuleFromSection(Section, *SectionName);
    break;
  }

  // If the kernel module is not found, this returns null and it is the caller's
  // responsibility to construct the kernel module by extracting per-TU modules
  // and cloning.
  return KernelModule;
}

void JitEngineDeviceHIP::extractModules(BinaryInfo &BinInfo) {
  Expected<object::ELF64LEFile> DeviceElf = getDeviceElf(BinInfo, DeviceArch);
  if (DeviceElf.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    PROTEUS_FATAL_ERROR("Error reading sections");

  SmallVector<std::unique_ptr<Module>> LinkedModules;

  auto ExtractModuleFromSection = [&DeviceElf, &BinInfo](
                                      auto &Section, StringRef SectionName) {
    ArrayRef<uint8_t> BitcodeData;
    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      PROTEUS_FATAL_ERROR("Error reading section contents");
    BitcodeData = *SectionContents;
    auto Bitcode = StringRef{reinterpret_cast<const char *>(BitcodeData.data()),
                             BitcodeData.size()};

    Timer T;
    SMDiagnostic Diag;
    auto M =
        getLazyIRModule(MemoryBuffer::getMemBufferCopy(Bitcode, SectionName),
                        Diag, *BinInfo.getLLVMContext(), true);
    if (!M)
      PROTEUS_FATAL_ERROR("Error parsing IR: " + Diag.getMessage());
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Parse IR " << SectionName << " " << T.elapsed()
                         << " ms\n");

    return M;
  };

  // We extract bitcode from sections. If there is a .jit.bitcode.lto section
  // due to RDC compilation, that will contain the fully-linked module created
  // during the AOT LTO pass, which is all we need.
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      PROTEUS_FATAL_ERROR("Error reading section name");
    PROTEUS_DBG(Logger::logs("proteus")
                << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto M = ExtractModuleFromSection(Section, *SectionName);

    LinkedModules.push_back(std::move(M));
  }

  BinInfo.setExtractedModules(LinkedModules);
}

hipFunction_t
JitEngineDeviceHIP::getKernelFunctionFromImage(StringRef KernelName,
                                               const void *Image) {
  return proteus::getKernelFunctionFromImage(
      KernelName, Image, Config::get().ProteusRelinkGlobalsByCopy,
      VarNameToDevPtr);
}

hipError_t JitEngineDeviceHIP::launchKernelFunction(hipFunction_t KernelFunc,
                                                    dim3 GridDim, dim3 BlockDim,
                                                    void **KernelArgs,
                                                    uint64_t ShmemSize,
                                                    hipStream_t Stream) {
  return proteus::launchKernelFunction(KernelFunc, GridDim, BlockDim,
                                       KernelArgs, ShmemSize, Stream);
}

JitEngineDeviceHIP::JitEngineDeviceHIP() {
  hipDeviceProp_t DevProp;
  proteusHipErrCheck(hipGetDeviceProperties(&DevProp, 0));

  DeviceArch = DevProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));
}

std::unique_ptr<MemoryBuffer> JitEngineDeviceHIP::compileOnly(Module &M, bool DisableIROpt) {
  if (!DisableIROpt) {
    const auto &CGConfig = Config::get().getCGConfig();
    proteus::optimizeIR(M, DeviceArch, CGConfig.optLevel(),
                        CGConfig.codeGenOptLevel());
  } else {
    if (Config::get().ProteusTraceOutput >= 1)
      Logger::trace("[SkipOpt] Skipping default<O3> IR optimization\n");
  }
  const auto &CGConfig = Config::get().getCGConfig();
  auto DeviceObject = proteus::codegenObject(
      M, DeviceArch, GlobalLinkedBinaries, CGConfig.codeGenOption());
  return DeviceObject;
}
