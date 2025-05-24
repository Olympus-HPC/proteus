//===-- JitEngineDeviceCUDA.cpp -- JIT Engine Device for CUDA --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/JitEngineDeviceCUDA.hpp"
#include "proteus/CoreLLVM.hpp"
#include "proteus/JitEngineDevice.hpp"
#include "proteus/Utils.h"
#include "proteus/UtilsCUDA.h"

#include <cuda_runtime.h>
#include <sys/types.h>

using namespace proteus;
using namespace llvm;

void *JitEngineDeviceCUDA::resolveDeviceGlobalAddr(const void *Addr) {
  return proteus::resolveDeviceGlobalAddr(Addr);
}

JitEngineDeviceCUDA &JitEngineDeviceCUDA::instance() {
  static JitEngineDeviceCUDA Jit{};
  return Jit;
}

void JitEngineDeviceCUDA::extractLinkedBitcode(
    LLVMContext &Ctx, CUmodule &CUMod,
    SmallVector<std::unique_ptr<Module>> &LinkedModules,
    std::string &ModuleId) {
  PROTEUS_DBG(Logger::logs("proteus")
              << "extractLinkedBitcode " << ModuleId << "\n");

  if (!ModuleIdToFatBinary.count(ModuleId))
    PROTEUS_FATAL_ERROR("Expected to find module id " + ModuleId + " in map");

  CUdeviceptr DevPtr;
  size_t Bytes;
  proteusCuErrCheck(
      cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str()));

  SmallString<4096> DeviceBitcode;
  DeviceBitcode.reserve(Bytes);
  proteusCuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));

  Timer T;
  StringRef Bitcode{DeviceBitcode.data(), Bytes};
  SMDiagnostic Diag;
  // We copy the Bitcode data for lazy parsing.
  // TODO: It could be zero-copy if we use our own backing storage for
  // DeviceBitcode.
  auto M = getLazyIRModule(MemoryBuffer::getMemBufferCopy(Bitcode, ModuleId),
                           Diag, Ctx, true);
  if (!M)
    PROTEUS_FATAL_ERROR("Error parsing IR: " + Diag.getMessage());
  M->setModuleIdentifier(ModuleId);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus") << "Parse IR " << ModuleId << " "
                                               << T.elapsed() << " ms\n";)

  LinkedModules.push_back(std::move(M));
}

HashT JitEngineDeviceCUDA::getModuleHash(BinaryInfo &BinInfo) {
  if (BinInfo.hasModuleHash())
    return BinInfo.getModuleHash();

  CUmodule CUMod;
  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();
  if (!FatbinWrapper)
    PROTEUS_FATAL_ERROR("Expected FatbinWrapper in map");

  auto &LinkedModuleIds = BinInfo.getModuleIds();

  proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  auto ExtractLinkedBitcodeHash = [&CUMod](std::string &ModuleId) {
    CUdeviceptr DevPtr;
    size_t Bytes;
    proteusCuErrCheck(
        cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str()));

    SmallVector<char, 4096> DeviceBitcode;
    DeviceBitcode.reserve(Bytes);
    proteusCuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
    return hash(StringRef{DeviceBitcode.data(), Bytes});
  };

  for (auto &ModuleId : LinkedModuleIds)
    BinInfo.updateModuleHash(ExtractLinkedBitcodeHash(ModuleId));

  for (auto &ModuleId : GlobalLinkedModuleIds)
    BinInfo.updateModuleHash(ExtractLinkedBitcodeHash(ModuleId));

  proteusCuErrCheck(cuModuleUnload(CUMod));

  return BinInfo.getModuleHash();
}

std::unique_ptr<Module> JitEngineDeviceCUDA::extractKernelModule(
    BinaryInfo &BinInfo, StringRef KernelName, LLVMContext &Ctx) {
  // We do not support emitting per-kernel modules during CUDA compilation,
  // hence this returns null to trigger the fallback to extraction of per-TU
  // modules and cloning.
  return nullptr;
}

void JitEngineDeviceCUDA::extractModules(BinaryInfo &BinInfo) {
  CUmodule CUMod;

  FatbinWrapperT *FatbinWrapper = BinInfo.getFatbinWrapper();
  if (!FatbinWrapper)
    PROTEUS_FATAL_ERROR("Expected FatbinWrapper in map");

  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = *BinInfo.getLLVMContext();

  auto &LinkedModuleIds = BinInfo.getModuleIds();

  proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  for (auto &ModuleId : LinkedModuleIds)
    extractLinkedBitcode(Ctx, CUMod, LinkedModules, ModuleId);

  for (auto &ModuleId : GlobalLinkedModuleIds)
    extractLinkedBitcode(Ctx, CUMod, LinkedModules, ModuleId);

  proteusCuErrCheck(cuModuleUnload(CUMod));

  BinInfo.setExtractedModules(LinkedModules);
}

void JitEngineDeviceCUDA::setLaunchBoundsForKernel(Module &M, Function &F,
                                                   size_t GridSize,
                                                   int BlockSize) {
  proteus::setLaunchBoundsForKernel(M, F, GridSize, BlockSize);
}

CUfunction JitEngineDeviceCUDA::getKernelFunctionFromImage(StringRef KernelName,
                                                           const void *Image) {
  return proteus::getKernelFunctionFromImage(
      KernelName, Image, Config::get().ProteusRelinkGlobalsByCopy,
      VarNameToDevPtr);
}

cudaError_t
JitEngineDeviceCUDA::launchKernelFunction(CUfunction KernelFunc, dim3 GridDim,
                                          dim3 BlockDim, void **KernelArgs,
                                          uint64_t ShmemSize, CUstream Stream) {
  return proteus::launchKernelFunction(KernelFunc, GridDim, BlockDim,
                                       KernelArgs, ShmemSize, Stream);
}

JitEngineDeviceCUDA::JitEngineDeviceCUDA() {
  // Initialize CUDA and retrieve the compute capability, needed for later
  // operations.
  CUdevice CUDev;
  CUcontext CUCtx;

  proteusCuErrCheck(cuInit(0));

  CUresult CURes = cuCtxGetDevice(&CUDev);
  if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
    // TODO: is selecting device 0 correct?
    proteusCuErrCheck(cuDeviceGet(&CUDev, 0));

  proteusCuErrCheck(cuCtxGetCurrent(&CUCtx));
  if (!CUCtx)
    proteusCuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));

  int CCMajor;
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  int CCMinor;
  proteusCuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  DeviceArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

  PROTEUS_DBG(Logger::logs("proteus") << "CUDA Arch " << DeviceArch << "\n");
}
