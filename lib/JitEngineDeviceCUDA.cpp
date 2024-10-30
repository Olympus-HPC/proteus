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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>

#include "JitEngineDeviceCUDA.hpp"
#include "Utils.h"

using namespace proteus;
using namespace llvm;

void *JitEngineDeviceCUDA::resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  cudaErrCheck(cudaGetSymbolAddress(&DevPtr, Addr));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
}

JitEngineDeviceCUDA &JitEngineDeviceCUDA::instance() {
  static JitEngineDeviceCUDA Jit{};
  return Jit;
}

std::unique_ptr<MemoryBuffer> JitEngineDeviceCUDA::extractDeviceBitcode(
    StringRef KernelName, const char *Binary, size_t FatbinSize) {
  CUmodule CUMod;
  CUlinkState CULinkState = nullptr;
  CUdeviceptr DevPtr;
  size_t Bytes;
  std::string Symbol = Twine("__jit_bc_" + KernelName).str();

  // NOTE: loading a module OR getting the global fails if rdc compilation
  // is enabled. Try to use the linker interface to load the binary image.
  // If that fails too, abort.
  // TODO: detect rdc compilation in the ProteusJitPass, see
  // __cudaRegisterLinkedLibrary or __nv_relfatbin section existences.
  if (cuModuleLoadFatBinary(&CUMod, Binary) != CUDA_SUCCESS ||
      cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, Symbol.c_str()) ==
          CUDA_ERROR_NOT_FOUND) {
    cuErrCheck(cuLinkCreate(0, nullptr, nullptr, &CULinkState));
    cuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY,
                             (void *)Binary, FatbinSize, "", 0, 0, 0));
    void *BinOut;
    size_t BinSize;
    cuErrCheck(cuLinkComplete(CULinkState, &BinOut, &BinSize));
    cuErrCheck(cuModuleLoadFatBinary(&CUMod, BinOut));
  }

  cuErrCheck(cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, Symbol.c_str()));

  SmallString<4096> DeviceBitcode;
  DeviceBitcode.reserve(Bytes);
  cuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
#ifdef ENABLE_DEBUG
  {
    std::error_code EC;
    raw_fd_ostream OutBC(Twine("from-device-jit-" + KernelName + ".bc").str(),
                         EC);
    if (EC)
      FATAL_ERROR("Cannot open device memory jit file");
    OutBC << StringRef(DeviceBitcode.data(), Bytes);
    OutBC.close();
  }
#endif

  cuErrCheck(cuModuleUnload(CUMod));
  if (CULinkState)
    cuErrCheck(cuLinkDestroy(CULinkState));
  return MemoryBuffer::getMemBufferCopy(StringRef(DeviceBitcode.data(), Bytes));
}

void JitEngineDeviceCUDA::setLaunchBoundsForKernel(Module *M, Function *F,
                                                   int GridSize,
                                                   int BlockSize) {
  NamedMDNode *NvvmAnnotations = M->getNamedMetadata("nvvm.annotations");
  assert(NvvmAnnotations && "Expected non-null nvvm.annotations metadata");
  // TODO: fix hardcoded 1024 as the maximum, by reading device
  // properties.
  // TODO: set min GridSize.
  int MaxThreads = std::min(1024, BlockSize);
  Metadata *MDVals[] = {ConstantAsMetadata::get(F),
                        MDString::get(M->getContext(), "maxntidx"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(M->getContext()), MaxThreads))};
  NvvmAnnotations->addOperand(MDNode::get(M->getContext(), MDVals));
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

  cuErrCheck(cuModuleLoadData(&Mod, Image));
  cuErrCheck(cuModuleGetFunction(&KernelFunc, Mod, KernelName.str().c_str()));

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
  nvPTXCompilerErrCheck(
      nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
  std::string ArchOpt = ("--gpu-name=" + DeviceArch).str();
#if ENABLE_DEBUG
  const char *CompileOptions[] = {ArchOpt.c_str(), "--verbose"};
  size_t NumCompileOptions = 2;
#else
  const char *CompileOptions[] = {ArchOpt.c_str()};
  size_t NumCompileOptions = 1;
#endif
  nvPTXCompilerErrCheck(
      nvPTXCompilerCompile(PTXCompiler, NumCompileOptions, CompileOptions));
  nvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgramSize(PTXCompiler, &BinSize));
  auto ObjBuf = WritableMemoryBuffer::getNewUninitMemBuffer(BinSize);
  nvPTXCompilerErrCheck(
      nvPTXCompilerGetCompiledProgram(PTXCompiler, ObjBuf->getBufferStart()));
#if ENABLE_DEBUG
  {
    size_t LogSize;
    nvPTXCompilerErrCheck(nvPTXCompilerGetInfoLogSize(PTXCompiler, &LogSize));
    auto Log = std::make_unique<char[]>(LogSize);
    nvPTXCompilerErrCheck(nvPTXCompilerGetInfoLog(PTXCompiler, Log.get()));
    dbgs() << "=== nvPTXCompiler Log\n" << Log.get() << "\n";
  }
#endif
  nvPTXCompilerErrCheck(nvPTXCompilerDestroy(&PTXCompiler));

  return std::move(ObjBuf);
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

  cuErrCheck(cuInit(0));

  CUresult CURes = cuCtxGetDevice(&CUDev);
  if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
    // TODO: is selecting device 0 correct?
    cuErrCheck(cuDeviceGet(&CUDev, 0));

  cuErrCheck(cuCtxGetCurrent(&CUCtx));
  if (!CUCtx)
    cuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));

  int CCMajor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
  int CCMinor;
  cuErrCheck(cuDeviceGetAttribute(
      &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
  DeviceArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

  DBG(dbgs() << "CUDA Arch " << DeviceArch << "\n");
}
