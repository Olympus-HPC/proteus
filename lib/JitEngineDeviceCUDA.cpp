//===-- JitEngineDeviceCUDA.cpp -- JIT Engine Device for CUDA --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>

#include "JitEngineDevice.hpp"
#include "JitEngineDeviceCUDA.hpp"
#include "Utils.h"
#include "UtilsCUDA.h"
#include <cuda_runtime.h>
#include <sys/types.h>

using namespace proteus;
using namespace llvm;

void *JitEngineDeviceCUDA::resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  proteusCudaErrCheck(cudaGetSymbolAddress(&DevPtr, Addr));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
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
    FATAL_ERROR("Expected to find module id " + ModuleId + " in map");

  FatbinWrapper_t *ModuleFatBinWrapper = ModuleIdToFatBinary[ModuleId];

  CUdeviceptr DevPtr;
  size_t Bytes;
  proteusCuErrCheck(
      cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str()));

  SmallString<4096> DeviceBitcode;
  DeviceBitcode.reserve(Bytes);
  proteusCuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));

  SMDiagnostic Err;
  auto M =
      parseIR(MemoryBufferRef(StringRef(DeviceBitcode.data(), Bytes), ModuleId),
              Err, Ctx);
  if (!M)
    FATAL_ERROR("unexpected");

  LinkedModules.push_back(std::move(M));
}

Module &JitEngineDeviceCUDA::extractDeviceBitcode(StringRef KernelName,
                                                  void *Kernel) {
  CUmodule CUMod;
  CUdeviceptr DevPtr;
  size_t Bytes;

  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto &Ctx = getProteusLLVMCtx();
  if (!KernelToHandleMap.contains(Kernel))
    FATAL_ERROR("Expected Kernel in map");

  void *Handle = KernelToHandleMap[Kernel];
  if (!HandleToBinaryInfo.contains(Handle))
    FATAL_ERROR("Expected Handle in map");

  if (!JITKernelInfoMap.contains(Kernel))
    FATAL_ERROR("Expected a Kernel Descriptor to exist");

  auto &KInfo = JITKernelInfoMap[Kernel];

  if (KInfo.hasLinkedIR())
    return KInfo.getLinkedModule();

  FatbinWrapper_t *FatbinWrapper = HandleToBinaryInfo[Handle].FatbinWrapper;
  if (!FatbinWrapper)
    FATAL_ERROR("Expected FatbinWrapper in map");

  auto &LinkedModuleIds = HandleToBinaryInfo[Handle].LinkedModuleIds;

  proteusCuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  for (auto &ModuleId : LinkedModuleIds)
    extractLinkedBitcode(Ctx, CUMod, LinkedModules, ModuleId);

  for (auto &ModuleId : GlobalLinkedModuleIds)
    extractLinkedBitcode(Ctx, CUMod, LinkedModules, ModuleId);

  proteusCuErrCheck(cuModuleUnload(CUMod));

  auto JitModule = linkJitModule(KernelName, LinkedModules);

  // Update modules of all kernels in our map
  for (const auto &KV : KernelToHandleMap) {
    // All kernels included in this collection of modules will have an identical
    // non specialized IR file. Map all Kernels, to this generic IR file
    if (KV.second != Handle)
      continue;
    if (!JITKernelInfoMap.contains(KV.first))
      continue;

    JITKernelInfoMap[KV.first].setLinkedModule(*JitModule);
  }

  if (!KInfo.hasLinkedIR())
    FATAL_ERROR("Expected KernelInfo to have updated Linked Modules");

  addLinkedModule(std::move(JitModule));

  return KInfo.getLinkedModule();
}

void JitEngineDeviceCUDA::setLaunchBoundsForKernel(Module &M, Function &F,
                                                   size_t GridSize,
                                                   int BlockSize) {
  NamedMDNode *NvvmAnnotations = M.getNamedMetadata("nvvm.annotations");
  assert(NvvmAnnotations && "Expected non-null nvvm.annotations metadata");
  // TODO: fix hardcoded 1024 as the maximum, by reading device
  // properties.
  // TODO: set min GridSize.
  int MaxThreads = std::min(1024, BlockSize);
  auto *FuncMetadata = ConstantAsMetadata::get(&F);
  auto *MaxntidxMetadata = MDString::get(M.getContext(), "maxntidx");
  auto *MaxThreadsMetadata = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(M.getContext()), MaxThreads));

  // Replace if the metadata exists.
  for (auto *MetadataNode : NvvmAnnotations->operands()) {
    // Expecting 3 operands ptr, desc, i32 value.
    assert(MetadataNode->getNumOperands() == 3);

    auto *PtrMetadata = MetadataNode->getOperand(0).get();
    auto *DescMetadata = MetadataNode->getOperand(1).get();
    if (PtrMetadata == FuncMetadata && MaxntidxMetadata == DescMetadata) {
      MetadataNode->replaceOperandWith(2, MaxThreadsMetadata);
      return;
    }
  }

  // Otherwise create the metadata and insert.
  Metadata *MDVals[] = {FuncMetadata, MaxntidxMetadata, MaxThreadsMetadata};
  NvvmAnnotations->addOperand(MDNode::get(M.getContext(), MDVals));
}

cudaError_t JitEngineDeviceCUDA::cudaModuleLaunchKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                 blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
  return cudaGetLastError();
}

CUfunction JitEngineDeviceCUDA::getKernelFunctionFromImage(StringRef KernelName,
                                                           const void *Image) {
  CUfunction KernelFunc;
  CUmodule Mod;

  proteusCuErrCheck(cuModuleLoadData(&Mod, Image));
  if (Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY) {
    for (auto &[GlobalName, HostAddr] : VarNameToDevPtr) {
      CUdeviceptr Dptr;
      size_t Bytes;
      proteusCuErrCheck(
          cuModuleGetGlobal(&Dptr, &Bytes, Mod, (GlobalName + "$ptr").c_str()));

      void *DevPtr = resolveDeviceGlobalAddr(HostAddr);
      uint64_t PtrVal = (uint64_t)DevPtr;
      proteusCuErrCheck(cuMemcpyHtoD(Dptr, &PtrVal, Bytes));
    }
  }
  proteusCuErrCheck(
      cuModuleGetFunction(&KernelFunc, Mod, KernelName.str().c_str()));

  return KernelFunc;
}

cudaError_t
JitEngineDeviceCUDA::launchKernelFunction(CUfunction KernelFunc, dim3 GridDim,
                                          dim3 BlockDim, void **KernelArgs,
                                          uint64_t ShmemSize, CUstream Stream) {
  return cudaModuleLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                                BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                                Stream, KernelArgs, nullptr);
}

cudaError_t JitEngineDeviceCUDA::launchKernelDirect(void *KernelFunc,
                                                    dim3 GridDim, dim3 BlockDim,
                                                    void **KernelArgs,
                                                    uint64_t ShmemSize,
                                                    CUstream Stream) {
  return cudaLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                          Stream);
}

void JitEngineDeviceCUDA::codegenPTX(Module &M, StringRef DeviceArch,
                                     SmallVectorImpl<char> &PTXStr) {
  // TODO: It is possbile to use PTX directly through the CUDA PTX JIT
  // interface. Maybe useful if we can re-link globals using the CUDA API.
  // Check this reference for PTX JIT caching:
  // https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
  // Interesting env vars: CUDA_CACHE_DISABLE, CUDA_CACHE_MAXSIZE,
  // CUDA_CACHE_PATH, CUDA_FORCE_PTX_JIT.
  TIMESCOPE("Codegen PTX");
  auto TMExpected = createTargetMachine(M, DeviceArch);
  if (!TMExpected)
    FATAL_ERROR(toString(TMExpected.takeError()));

  std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM.get()));

  raw_svector_ostream PTXOS(PTXStr);
  TM->addPassesToEmitFile(PM, PTXOS, nullptr, CGFT_AssemblyFile,
                          /* DisableVerify */ false, MMIWP);

  PM.run(M);
}

std::unique_ptr<MemoryBuffer>
JitEngineDeviceCUDA::codegenObject(Module &M, StringRef DeviceArch) {
  TIMESCOPE("Codegen object");
  SmallVector<char, 4096> PTXStr;
  size_t BinSize;

  codegenPTX(M, DeviceArch, PTXStr);
  PTXStr.push_back('\0');

  nvPTXCompilerHandle PTXCompiler;
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
  std::string ArchOpt = ("--gpu-name=" + DeviceArch).str();
  std::string RDCOption = "";
  if (!GlobalLinkedBinaries.empty())
    RDCOption = "-c";
#if PROTEUS_ENABLE_DEBUG
  const char *CompileOptions[] = {ArchOpt.c_str(), "--verbose",
                                  RDCOption.c_str()};
  size_t NumCompileOptions = 2 + (RDCOption.empty() ? 0 : 1);
#else
  const char *CompileOptions[] = {ArchOpt.c_str(), RDCOption.c_str()};
  size_t NumCompileOptions = 1 + (RDCOption.empty() ? 0 : 1);
#endif
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerCompile(PTXCompiler, NumCompileOptions, CompileOptions));
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgramSize(PTXCompiler, &BinSize));
  auto ObjBuf = WritableMemoryBuffer::getNewUninitMemBuffer(BinSize);
  proteusNvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgram(PTXCompiler, ObjBuf->getBufferStart()));
#if PROTEUS_ENABLE_DEBUG
  {
    size_t LogSize;
    proteusNvPTXCompilerErrCheck(
        nvPTXCompilerGetInfoLogSize(PTXCompiler, &LogSize));
    auto Log = std::make_unique<char[]>(LogSize);
    proteusNvPTXCompilerErrCheck(
        nvPTXCompilerGetInfoLog(PTXCompiler, Log.get()));
    Logger::logs("proteus") << "=== nvPTXCompiler Log\n" << Log.get() << "\n";
  }
#endif
  proteusNvPTXCompilerErrCheck(nvPTXCompilerDestroy(&PTXCompiler));

  std::unique_ptr<MemoryBuffer> FinalObjBuf;
  if (!GlobalLinkedBinaries.empty()) {
    CUlinkState CULinkState;
    proteusCuErrCheck(cuLinkCreate(0, nullptr, nullptr, &CULinkState));
    for (auto *Ptr : GlobalLinkedBinaries) {
      // We do not know the size of the binary but the CUDA API just needs a
      // non-zero argument.
      proteusCuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY, Ptr,
                                      1, "", 0, 0, 0));
    }

    // Again using a non-zero argument, though we can get the size from the ptx
    // compiler.
    proteusCuErrCheck(cuLinkAddData(
        CULinkState, CU_JIT_INPUT_FATBINARY,
        static_cast<void *>(ObjBuf->getBufferStart()), 1, "", 0, 0, 0));

    void *BinOut;
    size_t BinSize;
    proteusCuErrCheck(cuLinkComplete(CULinkState, &BinOut, &BinSize));
    FinalObjBuf = std::move(MemoryBuffer::getMemBufferCopy(
        StringRef{static_cast<char *>(BinOut), BinSize}));
  } else {
    FinalObjBuf = std::move(ObjBuf);
  }

  return std::move(FinalObjBuf);
}

JitEngineDeviceCUDA::JitEngineDeviceCUDA() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

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
