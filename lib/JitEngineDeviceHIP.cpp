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

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "JitEngineDeviceHIP.hpp"
#include "Utils.h"

using namespace proteus;
using namespace llvm;

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

std::unique_ptr<MemoryBuffer> JitEngineDeviceHIP::extractDeviceBitcode(
    StringRef KernelName, const char *Binary, size_t FatbinSize) {
  constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t Pos = 0;
  StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
    FATAL_ERROR("Error missing magic string");
  Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

  auto Read8ByteIntLE = [](const char *S, size_t Pos) {
    return support::endian::read64le(S + Pos);
  };

  uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
  Pos += 8;
  DBG(dbgs() << "NumberOfbundles " << NumberOfBundles << "\n");

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

    DBG(dbgs() << "Offset " << Offset << "\n");
    DBG(dbgs() << "Size " << Size << "\n");
    DBG(dbgs() << "TripleSize " << TripleSize << "\n");
    DBG(dbgs() << "Triple " << Triple << "\n");

    if (!Triple.contains("amdgcn"))
      continue;

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

  // NOTE: We must extract the .jit sections per kernel to avoid linked
  // device libraries. Otherwise, the hiprtc linker complains that it
  // cannot link device libraries (working assumption).
  ArrayRef<uint8_t> DeviceBitcode;
  Twine TargetSection = ".jit." + KernelName;
  for (auto Section : *Sections) {
    auto SectionName = DeviceElf->getSectionName(Section);
    if (SectionName.takeError())
      FATAL_ERROR("Error reading section name");
    DBG(dbgs() << "SectionName " << *SectionName << "\n");
    DBG(dbgs() << "TargetSection " << TargetSection << "\n");
    if (!SectionName->equals(TargetSection.str()))
      continue;

    auto SectionContents = DeviceElf->getSectionContents(Section);
    if (SectionContents.takeError())
      FATAL_ERROR("Error reading section contents");

    DeviceBitcode = *SectionContents;
  }

  if (DeviceBitcode.empty())
    FATAL_ERROR("Error finding the device bitcode");

  return MemoryBuffer::getMemBufferCopy(
      StringRef(reinterpret_cast<const char *>(DeviceBitcode.data()),
                DeviceBitcode.size()));
}

void JitEngineDeviceHIP::setLaunchBoundsForKernel(Module *M, Function *F,
                                                  int GridSize, int BlockSize) {
  // TODO: fix calculation of launch bounds.
  // TODO: find maximum (hardcoded 1024) from device info.
  // TODO: Setting as 1, BlockSize to replicate launch bounds settings
  // Does setting it as BlockSize, BlockSize help?
  F->addFnAttr("amdgpu-flat-work-group-size",
               "1," + std::to_string(std::min(1024, BlockSize)));
  // TODO: find warp size (hardcoded 64) from device info.
  // int WavesPerEU = (GridSize * BlockSize) / 64 / 110 / 4 / 2;
  int WavesPerEU = 0;
  // F->addFnAttr("amdgpu-waves-per-eu", std::to_string(WavesPerEU));
  DBG(dbgs() << "BlockSize " << BlockSize << " GridSize " << GridSize
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