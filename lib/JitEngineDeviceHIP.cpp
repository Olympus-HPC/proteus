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
#include <string>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "JitEngineDeviceHIP.hpp"
#include "Utils.h"

using namespace proteus;
using namespace llvm;

static inline const std::string &getArch() {
  static std::string Arch{[]() {
    int Device;
    hipErrCheck(hipInit(0));
    hipErrCheck(hipGetDevice(&Device));
    hipDeviceProp_t DeviceProperties;

    // Get properties of the current device
    hipErrCheck(hipGetDeviceProperties(&DeviceProperties, Device));

    // Get the full architecture name (e.g., gfx90a:sramecc+:xnack-)
    std::string arch_name = DeviceProperties.gcnArchName;

    // Find the colon (:) to isolate the base architecture
    auto DevArch = std::string(arch_name.substr(0, arch_name.find(':')));
    DBG(Logger::logs("proteus")
        << "Device Architecture is " << DevArch << "\n");
    return DevArch;
  }()};

  return Arch;
}

void *JitEngineDeviceHIP::resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  hipErrCheck(hipGetSymbolAddress(&DevPtr, HIP_SYMBOL(Addr)));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
}

JitEngineDeviceHIP &JitEngineDeviceHIP::instance() {
  static JitEngineDeviceHIP Jit{};
  return Jit;
}

Module &JitEngineDeviceHIP::extractDeviceBitcode(StringRef KernelName,
                                                 void *Kernel) {
  constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t Pos = 0;

  if (!KernelToHandleMap.contains(Kernel))
    FATAL_ERROR("Expected Kernel in map");

  if (!JITKernelInfoMap.contains(Kernel))
    FATAL_ERROR("Expected a Kernel Descriptor to exist");

  auto &KInfo = JITKernelInfoMap[Kernel];

  if (KInfo.hasLinkedIR())
    return KInfo.getLinkedModule();

  void *Handle = KernelToHandleMap[Kernel];
  if (!HandleToBinaryInfo.contains(Handle))
    FATAL_ERROR("Expected Handle in map");

  FatbinWrapper_t *FatbinWrapper = HandleToBinaryInfo[Handle].FatbinWrapper;
  if (!FatbinWrapper)
    FATAL_ERROR("Expected FatbinWrapper in map");

  const char *Binary = FatbinWrapper->Binary;

  StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
    FATAL_ERROR("Error missing magic string");
  Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

  auto Read8ByteIntLE = [](const char *S, size_t Pos) {
    return support::endian::read64le(S + Pos);
  };

  uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
  Pos += 8;
  DBG(Logger::logs("proteus") << "NumberOfbundles " << NumberOfBundles << "\n");

  StringRef DeviceBinary;
  for (uint64_t i = 0; i < NumberOfBundles; ++i) {
    uint64_t Offset = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    uint64_t Size = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    uint64_t TripleSize = Read8ByteIntLE(Binary, Pos);
    Pos += 8;

    StringRef Triple(Binary + Pos, TripleSize);
    Pos += TripleSize;

    DBG(Logger::logs("proteus") << "Offset " << Offset << "\n");
    DBG(Logger::logs("proteus") << "Size " << Size << "\n");
    DBG(Logger::logs("proteus") << "TripleSize " << TripleSize << "\n");
    DBG(Logger::logs("proteus") << "Triple " << Triple << "\n");

    if (!Triple.contains("amdgcn") || !Triple.contains(getArch())) {
      DBG(Logger::logs("proteus")
          << "Miss-matching architecture, skipping ...\n");
      continue;
    }

    DeviceBinary = StringRef(Binary + Offset, Size);
    break;
  }

  Expected<object::ELF64LEFile> DeviceElf =
      object::ELF64LEFile::create(DeviceBinary);
  if (DeviceElf.takeError())
    FATAL_ERROR("Cannot create the device elf");

  auto Sections = DeviceElf->sections();
  if (Sections.takeError())
    FATAL_ERROR("Error reading sections");

  ArrayRef<uint8_t> DeviceBitcode;
  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = getProteusLLVMCtx();

  auto extractModuleFromSection = [&DeviceElf, &Ctx](auto &Section,
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
    DBG(Logger::logs("proteus") << "SectionName " << *SectionName << "\n");

    if (!SectionName->starts_with(".jit.bitcode"))
      continue;

    auto M = extractModuleFromSection(Section, *SectionName);

    if (SectionName->equals(".jit.bitcode.lto")) {
      LinkedModules.clear();
      LinkedModules.push_back(std::move(M));
      break;
    } else {
      LinkedModules.push_back(std::move(M));
    }
  }

  auto JitModule = linkJitModule(KernelName, LinkedModules);

  // All kernels included in this collection of modules will have an
  // identical non specialized IR file. Map all Kernels, to this generic IR
  // file
  [this, &JitModule, &Handle]() {
    DenseSet<StringRef> KernelNames;
    for (auto &Func : *JitModule) {
      if (Func.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
        KernelNames.insert(Func.getName());
      }
    }

    for (const auto &KV : KernelToHandleMap) {

      if (KV.second != Handle)
        continue;

      // This is likely a not required check.
      // KV.first is in KernelToHandleMap, so we should have the Descriptor.
      if (!JITKernelInfoMap.contains(KV.first))
        continue;

      auto &KDescr = JITKernelInfoMap[KV.first];
      if (!KernelNames.contains(KDescr.getName()))
        continue;

      KDescr.setLinkedModule(*JitModule);
    }
  }();

  if (!KInfo.hasLinkedIR())
    FATAL_ERROR("Expected KernelInfo to have updated Linked Modules");

  addLinkedModule(std::move(JitModule));
  return KInfo.getLinkedModule();
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
  DBG(Logger::logs("proteus")
      << "BlockSize " << BlockSize << " GridSize " << GridSize
      << " => Set Wokgroup size " << BlockSize << " WavesPerEU (unused) "
      << WavesPerEU << "\n");
}

std::unique_ptr<MemoryBuffer>
JitEngineDeviceHIP::codegenObject(Module &M, StringRef DeviceArch) {
  TIMESCOPE("Codegen object");
  char *BinOut;
  size_t BinSize;

  SmallString<4096> ModuleBuf;
  raw_svector_ostream ModuleBufOS(ModuleBuf);
  WriteBitcodeToFile(M, ModuleBufOS);

  hiprtcLinkState hip_link_state_ptr;

  // NOTE: This code is an example of passing custom, AMD-specific
  // options to the compiler/linker.
  // NOTE: Unrolling can have a dramatic (time-consuming) effect on JIT
  // compilation time and on the resulting optimization, better or worse
  // depending on code specifics.
  std::string MArchOpt = ("-march=" + DeviceArch).str();
  const char *OptArgs[] = {"-mllvm", "-amdgpu-internalize-symbols", "-mllvm",
                           "-unroll-threshold=1000", MArchOpt.c_str()};
  std::vector<hiprtcJIT_option> JITOptions = {
      HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t OptArgsSize = 5;
  const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
  hiprtcErrCheck(hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                                  (void **)JITOptionsValues,
                                  &hip_link_state_ptr));
  // NOTE: the following version of te code does not set options.
  // hiprtcErrCheck(hiprtcLinkCreate(0, nullptr, nullptr, &hip_link_state_ptr));

  hiprtcErrCheck(hiprtcLinkAddData(
      hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
      (void *)ModuleBuf.data(), ModuleBuf.size(), "", 0, nullptr, nullptr));
  hiprtcErrCheck(
      hiprtcLinkComplete(hip_link_state_ptr, (void **)&BinOut, &BinSize));

  return MemoryBuffer::getMemBuffer(StringRef{BinOut, BinSize});
}

hipFunction_t
JitEngineDeviceHIP::getKernelFunctionFromImage(StringRef KernelName,
                                               const void *Image) {
  hipModule_t HipModule;
  hipFunction_t KernelFunc;

  hipErrCheck(hipModuleLoadData(&HipModule, Image));
  hipErrCheck(
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
  hipErrCheck(hipGetDeviceProperties(&devProp, 0));

  DeviceArch = devProp.gcnArchName;
  DeviceArch = DeviceArch.substr(0, DeviceArch.find_first_of(":"));
}

// === APPENDIX ===
// TODO: Dynamic linking is to be supported through hiprtc. Currently
// the interface is limited and lacks support for linking globals.
// Indicative code here is for future re-visit.
#if DYNAMIC_LINK
std::vector<hiprtcJIT_option> LinkOptions = {HIPRTC_JIT_GLOBAL_SYMBOL_NAMES,
                                             HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS,
                                             HIPRTC_JIT_GLOBAL_SYMBOL_COUNT};
std::vector<const char *> GlobalNames;
std::vector<const void *> GlobalAddrs;
for (auto RegisterVar : VarNameToDevPtr) {
  auto &VarName = RegisterVar.first;
  auto DevPtr = RegisterVar.second;
  GlobalNames.push_back(VarName.c_str());
  GlobalAddrs.push_back(DevPtr);
}

std::size_t GlobalSize = GlobalNames.size();
std::size_t NumOptions = LinkOptions.size();
const void *LinkOptionsValues[] = {GlobalNames.data(), GlobalAddrs.data(),
                                   (void *)&GlobalSize};
hiprtcErrCheck(hiprtcLinkCreate(LinkOptions.size(), LinkOptions.data(),
                                (void **)&LinkOptionsValues,
                                &hip_link_state_ptr));

hiprtcErrCheck(hiprtcLinkAddData(
    hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
    (void *)ModuleBuffer.data(), ModuleBuffer.size(), KernelName.data(),
    LinkOptions.size(), LinkOptions.data(), (void **)&LinkOptionsValues));
#endif
