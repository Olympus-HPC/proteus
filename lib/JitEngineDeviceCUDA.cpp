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
  cudaErrCheck(cudaGetSymbolAddress(&DevPtr, Addr));
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
  DBG(dbgs() << "extractLinkedBitcode " << ModuleId << "\n");

  if (!ModuleIdToFatBinary.count(ModuleId))
    FATAL_ERROR("Expected to find module id " + ModuleId + " in map");

  FatbinWrapper_t *ModuleFatBinWrapper = ModuleIdToFatBinary[ModuleId];
  // Remove proteus-compiled binary with JIT bitcode from linked binaries.
  GlobalLinkedBinaries.erase((void *)ModuleFatBinWrapper->Binary);

  CUdeviceptr DevPtr;
  size_t Bytes;
  cuErrCheck(cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, ModuleId.c_str()));

  SmallString<4096> DeviceBitcode;
  DeviceBitcode.reserve(Bytes);
  cuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));

  SMDiagnostic Err;
  auto M =
      parseIR(MemoryBufferRef(StringRef(DeviceBitcode.data(), Bytes), ModuleId),
              Err, Ctx);
  if (!M)
    FATAL_ERROR("unexpected");

  LinkedModules.push_back(std::move(M));
}

std::unique_ptr<MemoryBuffer>
JitEngineDeviceCUDA::extractDeviceBitcode(StringRef KernelName, void *Kernel) {
  CUmodule CUMod;
  CUdeviceptr DevPtr;
  size_t Bytes;

  SmallVector<std::unique_ptr<Module>> LinkedModules;
  auto Ctx = std::make_unique<LLVMContext>();
  auto JitModule = std::make_unique<llvm::Module>("JitModule", *Ctx);
  if (!KernelToHandleMap.contains(Kernel))
    FATAL_ERROR("Expected Kernel in map");

  void *Handle = KernelToHandleMap[Kernel];
  if (!HandleToBinaryInfo.contains(Handle))
    FATAL_ERROR("Expected Handle in map");

  FatbinWrapper_t *FatbinWrapper = HandleToBinaryInfo[Handle].FatbinWrapper;
  if (!FatbinWrapper)
    FATAL_ERROR("Expected FatbinWrapper in map");

  auto &LinkedModuleIds = HandleToBinaryInfo[Handle].LinkedModuleIds;

  cuErrCheck(cuModuleLoadData(&CUMod, FatbinWrapper->Binary));

  for (auto &ModuleId : LinkedModuleIds)
    extractLinkedBitcode(*Ctx.get(), CUMod, LinkedModules, ModuleId);

  for (auto &ModuleId : GlobalLinkedModuleIds)
    extractLinkedBitcode(*Ctx.get(), CUMod, LinkedModules, ModuleId);

  cuErrCheck(cuModuleUnload(CUMod));

  linkJitModule(JitModule.get(), Ctx.get(), KernelName, LinkedModules);

  std::string LinkedDeviceBitcode;
  raw_string_ostream OS(LinkedDeviceBitcode);
  WriteBitcodeToFile(*JitModule.get(), OS);
  OS.flush();

  return MemoryBuffer::getMemBufferCopy(LinkedDeviceBitcode);
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
  Metadata *MDVals[] = {ConstantAsMetadata::get(&F),
                        MDString::get(M.getContext(), "maxntidx"),
                        ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(M.getContext()), MaxThreads))};
  NvvmAnnotations->addOperand(MDNode::get(M.getContext(), MDVals));
}

void JitEngineDeviceCUDA::setKernelDims(Module &M, Function &F, dim3 &GridDim,
                                        dim3 &BlockDim) {
  auto ReplaceIntrinsicDim = [&](StringRef IntrinsicName, uint32_t value) {
    Function *IntrinsicFunction = M.getFunction(IntrinsicName);
    if (!IntrinsicFunction)
      return;

    for (auto U = IntrinsicFunction->use_begin(),
              UE = IntrinsicFunction->use_end();
         U != UE;) {
      Use &Use = *U++;
      if (auto *Call = dyn_cast<CallInst>(Use.getUser())) {
        Value *ConstantValue =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), value);
        Call->replaceAllUsesWith(ConstantValue);
        Call->eraseFromParent();
      }
    }
  };
  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.nctaid.x", GridDim.x);
  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.nctaid.y", GridDim.y);
  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.nctaid.z", GridDim.z);

  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.ntid.z", BlockDim.x);
  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.ntid.y", BlockDim.y);
  ReplaceIntrinsicDim("llvm.nvvm.read.ptx.sreg.ntid.z", BlockDim.z);

  auto InsertAssume = [&](StringRef IntrinsicName, int BlockDim) {
    Function *IntrinsicFunction = M.getFunction(IntrinsicName);
    if (!IntrinsicFunction || IntrinsicFunction->use_empty())
      return; // No modifications made if the intrinsic is not used

    // Iterate over all uses of the intrinsic
    for (auto U = IntrinsicFunction->use_begin(),
              UE = IntrinsicFunction->use_end();
         U != UE;) {
      Use &Use = *U++;
      if (auto *Call = dyn_cast<CallInst>(Use.getUser())) {
        // Insert the llvm.assume intrinsic
        IRBuilder<> Builder(Call->getNextNode());
        Value *Bound = ConstantInt::get(Call->getType(), BlockDim);
        Value *Cmp = Builder.CreateICmpULT(Call, Bound);

        Function *AssumeIntrinsic =
            Intrinsic::getDeclaration(&M, Intrinsic::assume);
        Builder.CreateCall(AssumeIntrinsic, Cmp);
      }
    }
  };

  InsertAssume("llvm.nvvm.read.ptx.sreg.tid.x", BlockDim.x);
  InsertAssume("llvm.nvvm.read.ptx.sreg.tid.y", BlockDim.y);
  InsertAssume("llvm.nvvm.read.ptx.sreg.tid.z", BlockDim.z);
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
  CUlinkState CULinkState = nullptr;

  const void *Binary = Image;
  if (!GlobalLinkedBinaries.empty()) {
    cuErrCheck(cuLinkCreate(0, nullptr, nullptr, &CULinkState));
    for (auto *Ptr : GlobalLinkedBinaries)
      // We do not know the size of the binary but the CUDA API just needs a
      // non-zero argument.
      cuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY, Ptr, 1, "",
                               0, 0, 0));

    // Again using a non-zero argument, though we can get the size from the ptx
    // compiler.
    cuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY,
                             const_cast<void *>(Image), 1, "", 0, 0, 0));

    void *BinOut;
    size_t BinSize;
    cuErrCheck(cuLinkComplete(CULinkState, &BinOut, &BinSize));
    Binary = BinOut;
  }

  cuErrCheck(cuModuleLoadData(&Mod, Binary));
  cuErrCheck(cuModuleGetFunction(&KernelFunc, Mod, KernelName.str().c_str()));

  if (CULinkState)
    cuLinkDestroy(CULinkState);

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
  nvPTXCompilerErrCheck(
      nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
  std::string ArchOpt = ("--gpu-name=" + DeviceArch).str();
  std::string RDCOption = "";
  if (!GlobalLinkedBinaries.empty())
    RDCOption = "-c";
#if ENABLE_DEBUG
  const char *CompileOptions[] = {ArchOpt.c_str(), "--verbose",
                                  RDCOption.c_str()};
  size_t NumCompileOptions = 2 + (RDCOption.empty() ? 0 : 1);
#else
  const char *CompileOptions[] = {ArchOpt.c_str(), RDCOption.c_str()};
  size_t NumCompileOptions = 1 + (RDCOption.empty() ? 0 : 1);
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
