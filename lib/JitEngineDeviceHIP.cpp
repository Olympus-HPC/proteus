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
    return llvm::support::endian::read64le(S + Pos);
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

#if ENABLE_DEBUG
  {
    std::error_code EC;
    raw_fd_ostream OutBin("device.bin", EC);
    if (EC)
      FATAL_ERROR("Cannot open device binary file");
    OutBin << DeviceBinary;
    OutBin.close();
    dbgs() << "Binary image found\n";
  }
#endif

  Expected<object::ELF64LEFile> DeviceElf =
      llvm::object::ELF64LEFile::create(DeviceBinary);
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

#if ENABLE_DEBUG
  {
    std::error_code EC;
    raw_fd_ostream OutBC(Twine(".jit." + KernelName + ".bc").str(), EC);
    if (EC)
      FATAL_ERROR("Cannot open device bitcode file");
    OutBC << StringRef(reinterpret_cast<const char *>(DeviceBitcode.data()),
                       DeviceBitcode.size());
    OutBC.close();
  }
#endif

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

hipError_t JitEngineDeviceHIP::codegenAndLaunch(
    Module *M, StringRef HipArch, StringRef KernelName, StringRef Suffix,
    uint64_t HashValue, RuntimeConstant *RC, int NumRuntimeConstants,
    dim3 GridDim, dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
    hipStream_t Stream) {
  // Remove extras to get a working CPU architecture value, e.g., from
  // gfx90a:sramecc+:xnack- drop everything after the first :.
  // HipArch = HipArch.substr(0, HipArch.find_first_of(":"));
  // TODO: Do not run optimization pipeline for hip, hiprtc adds O3 by
  // default. Also, need to fine-tune the pipeline: issue with libor where
  // aggressive unrolling creates huge, slow binary code.
  // runOptimizationPassPipeline(*M, HipArch);
#if ENABLE_DEBUG
  {
    if (verifyModule(*M, &errs()))
      FATAL_ERROR("Broken module found after optimization, JIT "
                  "compilation aborted!");
    std::error_code EC;
    raw_fd_ostream OutBC(
        Twine("opt-transformed-jit-" + KernelName + Suffix + ".bc").str(), EC);
    if (EC)
      FATAL_ERROR("Cannot open device transformed bitcode file");
    // TODO: Remove or leave it only for debugging.
    OutBC << *M;
    OutBC.close();
  }
#endif

  SmallString<4096> ModuleBuffer;
  raw_svector_ostream ModuleBufferOS(ModuleBuffer);
  WriteBitcodeToFile(*M, ModuleBufferOS);

  char *BinOut;
  size_t BinSize;
  hipModule_t HipModule;

  hiprtcLinkState hip_link_state_ptr;

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

  {
    TIMESCOPE("Device linker");
// #if CUSTOM_OPTIONS
// TODO: Toggle this with an env var.
#if 1
    // NOTE: This code is an example of passing custom, AMD-specific
    // options to the compiler/linker. NOTE: Unrolling can have a dramatic
    // (time-consuming) effect on JIT compilation time and on the
    // resulting optimization, better or worse depending on code
    // specifics. const char *OptArgs[] = {"-mllvm",
    // "-amdgpu-internalize-symbols",
    //                         "-save-temps", "-mllvm",
    //                         "-unroll-threshold=100"};
    const char *OptArgs[] = {"-mllvm", "-amdgpu-internalize-symbols", "-mllvm",
                             "-unroll-threshold=1000", "-march=gfx90a"};
    std::vector<hiprtcJIT_option> JITOptions = {
        HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
    size_t OptArgsSize = 5;
    const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
    hiprtcErrCheck(hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                                    (void **)JITOptionsValues,
                                    &hip_link_state_ptr));
#else
    hiprtcErrCheck(hiprtcLinkCreate(0, nullptr, nullptr, &hip_link_state_ptr));
#endif

    hiprtcErrCheck(
        hiprtcLinkAddData(hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
                          (void *)ModuleBuffer.data(), ModuleBuffer.size(),
                          KernelName.data(), 0, nullptr, nullptr));
    hiprtcErrCheck(
        hiprtcLinkComplete(hip_link_state_ptr, (void **)&BinOut, &BinSize));
  }
  {
    TIMESCOPE("Load module");
    hipErrCheck(hipModuleLoadData(&HipModule, BinOut));
  }

  hipFunction_t HipFunction;
  {
    TIMESCOPE("Module get function");
    hipErrCheck(hipModuleGetFunction(&HipFunction, HipModule,
                                     (KernelName + Suffix).str().c_str()));
  }
  CodeCache.insert(HashValue, HipFunction, KernelName, RC, NumRuntimeConstants);

  StringRef ObjectRef(BinOut, BinSize);
  if (Config.ENV_JIT_USE_STORED_CACHE)
    StoredCache.storeObject(HashValue, ObjectRef);

  hipErrCheck(hipModuleLaunchKernel(
      HipFunction, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
      BlockDim.z, ShmemSize, (hipStream_t)Stream, KernelArgs, nullptr));
  return hipSuccess;
}

hipError_t JitEngineDeviceHIP::compileAndRun(
    StringRef ModuleUniqueId, StringRef KernelName,
    FatbinWrapper_t *FatbinWrapper, size_t FatbinSize, RuntimeConstant *RC,
    int NumRuntimeConstants, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, void *Stream) {
  TIMESCOPE("compileAndRun");

  uint64_t HashValue =
      CodeCache.hash(ModuleUniqueId, KernelName, RC, NumRuntimeConstants);

  std::string Suffix = mangleSuffix(HashValue);

  hipFunction_t HipFunction = CodeCache.lookup(HashValue);
  if (HipFunction)
    return hipModuleLaunchKernel(HipFunction, GridDim.x, GridDim.y, GridDim.z,
                                 BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                                 (hipStream_t)Stream, KernelArgs, nullptr);

  if (Config.ENV_JIT_USE_STORED_CACHE)
    if ((HipFunction = StoredCache.lookup(
             HashValue, (KernelName + Suffix).str(),
             [](StringRef Filename, StringRef Kernel) -> hipFunction_t {
               hipFunction_t DevFunction;
               hipModule_t HipModule;
               //  Load module from file.
               auto Err = hipModuleLoad(&HipModule, Filename.str().c_str());
               if (Err == hipErrorFileNotFound)
                 return nullptr;

               hipErrCheck(hipModuleGetFunction(&DevFunction, HipModule,
                                                Kernel.str().c_str()));

               return DevFunction;
             }

             ))) {
      CodeCache.insert(HashValue, HipFunction, KernelName, RC,
                       NumRuntimeConstants);

      return hipModuleLaunchKernel(
          HipFunction, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
          BlockDim.z, ShmemSize, (hipStream_t)Stream, KernelArgs, nullptr);
    }

  hipDeviceProp_t devProp;
  hipErrCheck(hipGetDeviceProperties(&devProp, 0));
  auto IRBuffer = extractDeviceBitcode(KernelName, FatbinWrapper->Binary);

  auto TransformedBitcode = specializeBitcode(
      KernelName, Suffix, IRBuffer->getBuffer(),
      BlockDim.x * BlockDim.y * BlockDim.z, GridDim.x * GridDim.y * GridDim.z,
      RC, NumRuntimeConstants);
  if (auto E = TransformedBitcode.takeError())
    FATAL_ERROR(toString(std::move(E)).c_str());

  Module *M = TransformedBitcode->getModuleUnlocked();

#if ENABLE_DEBUG
  {
    std::error_code EC;
    raw_fd_ostream OutBC(
        Twine("transformed-jit-" + KernelName + Suffix + ".bc").str(), EC);
    if (EC)
      FATAL_ERROR("Cannot open device transformed bitcode file");
    OutBC << *M;
    OutBC.close();
  }
#endif

  auto Ret =
      codegenAndLaunch(M, devProp.gcnArchName, KernelName, Suffix, HashValue,
                       RC, NumRuntimeConstants, GridDim, BlockDim, KernelArgs,
                       ShmemSize, (hipStream_t)Stream);

  return Ret;
}

JitEngineDeviceHIP::JitEngineDeviceHIP() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
}
