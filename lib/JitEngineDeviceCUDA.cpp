//===-- JitEngineDeviceCUDA.cpp -- JIT Engine Device for CUDA --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

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
  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F),
      llvm::MDString::get(M->getContext(), "maxntidx"),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(M->getContext()), MaxThreads))};
  NvvmAnnotations->addOperand(llvm::MDNode::get(M->getContext(), MDVals));
}

void JitEngineDeviceCUDA::codegenPTX(Module &M, StringRef CudaArch,
                                     SmallVectorImpl<char> &PTXStr) {
  TIMESCOPE("Codegen PTX");
  auto TMExpected = createTargetMachine(M, CudaArch);
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

cudaError_t JitEngineDeviceCUDA::codegenAndLaunch(
    Module *M, StringRef CudaArch, StringRef KernelName, StringRef Suffix,
    uint64_t HashValue, RuntimeConstant *RC, int NumRuntimeConstants,
    dim3 GridDim, dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
    CUstream Stream) {
  // Codegen PTX, load the module and run through the CUDA PTX JIT
  // interface. Check this reference for JIT caching:
  // https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
  // Interesting env vars: CUDA_CACHE_DISABLE, CUDA_CACHE_MAXSIZE,
  // CUDA_CACHE_PATH, CUDA_FORCE_PTX_JIT.
  // For CUDA, run the target-specific optimization pipeline to optimize the
  // LLVM IR before handing over to the CUDA driver PTX compiler.
  runOptimizationPassPipeline(*M, CudaArch);

  SmallVector<char, 4096> PTXStr;
  SmallVector<char, 4096> FinalIR;
  size_t BinSize;

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
    OutBC << *M;
    OutBC.close();
  }
#endif

  codegenPTX(*M, CudaArch, PTXStr);

#if ENABLE_DEBUG
  {
    std::error_code EC;
    raw_fd_ostream OutPtx(
        Twine("jit-" + std::to_string(HashValue) + ".ptx").str(), EC);
    if (EC)
      FATAL_ERROR("Cannot open ptx output file");
    OutPtx << PTXStr;
    OutPtx.close();
  }
#endif

  CUmodule CUMod;
  CUfunction CUFunc;

  {
    TIMESCOPE("Create object");
    // CUDA requires null-terminated PTX.
    PTXStr.push_back('\0');
#if ENABLE_LLVMIR_STORED_CACHE
    {
      raw_svector_ostream IROS(FinalIR);
      WriteBitcodeToFile(*M, IROS);
    }
    StringRef ObjectRef(FinalIR.data(), FinalIR.size());
#elif ENABLE_CUDA_PTX_STORED_CACHE
    cuErrCheck(cuModuleLoadData(&CUMod, PTXStr.data()));
    cuErrCheck(cuModuleGetFunction(&CUFunc, CUMod,
                                   Twine(KernelName + Suffix).str().c_str()));
    StringRef ObjectRef(PTXStr.data(), PTXStr.size());
#else
    // Store ELF object.
    nvPTXCompilerHandle PTXCompiler;
    nvPTXCompilerErrCheck(
        nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
    std::string ArchOpt = ("--gpu-name=" + CudaArch).str();
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
    auto BinOut = std::make_unique<char[]>(BinSize);
    nvPTXCompilerErrCheck(
        nvPTXCompilerGetCompiledProgram(PTXCompiler, BinOut.get()));

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

    cuErrCheck(cuModuleLoadData(&CUMod, BinOut.get()));
    cuErrCheck(cuModuleGetFunction(&CUFunc, CUMod,
                                   Twine(KernelName + Suffix).str().c_str()));

    StringRef ObjectRef(BinOut.get(), BinSize);
#endif

    CodeCache.insert(HashValue, CUFunc, KernelName, RC, NumRuntimeConstants);

    if (Config.ENV_JIT_USE_STORED_CACHE)
      StoredCache.storeObject(HashValue, ObjectRef);
  }

  cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                 BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                 KernelArgs, nullptr);
  // TODO: cuModuleUnload and ctxCtxDestroy at program exit.
  return cudaGetLastError();
}

cudaError_t JitEngineDeviceCUDA::compileAndRun(
    StringRef ModuleUniqueId, StringRef KernelName,
    FatbinWrapper_t *FatbinWrapper, size_t FatbinSize, RuntimeConstant *RC,
    int NumRuntimeConstants, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, void *Stream) {
  TIMESCOPE("compileAndRun");

  uint64_t HashValue =
      CodeCache.hash(ModuleUniqueId, KernelName, RC, NumRuntimeConstants);
  // NOTE: we don't need a suffix to differentiate kernels, each
  // specialization will be in its own module uniquely identify by HashValue. It
  // exists only for debugging purposes to verify that the jitted kernel
  // executes.
  std::string Suffix = mangleSuffix(HashValue);

  CUfunction CUFunc = CodeCache.lookup(HashValue);
  if (CUFunc) {
    cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                   BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                   KernelArgs, nullptr);
    return cudaGetLastError();
  }

  if (Config.ENV_JIT_USE_STORED_CACHE)
    if ((CUFunc = StoredCache.lookup(
             HashValue, (KernelName + Suffix).str(),
             [](StringRef Filename, StringRef Kernel) -> CUfunction {
               CUfunction DevFunction;
               CUmodule CUMod;
               auto Err = cuModuleLoad(&CUMod, Filename.str().c_str());

               if (Err == CUDA_ERROR_FILE_NOT_FOUND)
                 return nullptr;

               cuErrCheck(Err);

               cuErrCheck(cuModuleGetFunction(&DevFunction, CUMod,
                                              Kernel.str().c_str()));

               return DevFunction;
             }))) {
      CodeCache.insert(HashValue, CUFunc, KernelName, RC, NumRuntimeConstants);

      cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                     BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                     KernelArgs, nullptr);
      return cudaGetLastError();
    }

  auto IRBuffer =
      extractDeviceBitcode(KernelName, FatbinWrapper->Binary, FatbinSize);

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

  auto Ret = codegenAndLaunch(M, CudaArch, KernelName, Suffix, HashValue, RC,
                              NumRuntimeConstants, GridDim, BlockDim,
                              KernelArgs, ShmemSize, (CUstream)Stream);
  return Ret;
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
  CudaArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

  DBG(dbgs() << "CUDA Arch " << CudaArch << "\n");
}
