#ifndef PROTEUS_CORE_CUDA_H
#define PROTEUS_CORE_CUDA_H

#include "proteus/Error.h"
#include "proteus/impl/GlobalVarInfo.h"
#include "proteus/impl/UtilsCUDA.h"

#include <llvm/ADT/StringRef.h>

#include <unordered_map>

namespace proteus {

extern "C" {
// Definitions for function pointers to CUDA APIs that we will resolve at
// runtime using builtins.
// NOLINTBEGIN(readability-identifier-naming)
inline cudaError_t (*__proteus_cudaGetSymbolAddress_ptr)(
    void **, const void *) = nullptr;
inline cudaError_t (*__proteus_cudaLaunchKernel_ptr)(const void *, dim3, dim3,
                                                     void **, size_t,
                                                     cudaStream_t) = nullptr;
}
// NOLINTEND(readability-identifier-naming)

inline void *resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  if (__proteus_cudaGetSymbolAddress_ptr) {
    __proteus_cudaGetSymbolAddress_ptr(&DevPtr, Addr);
    assert(DevPtr && "Expected non-null device pointer for global");
    return DevPtr;
  }

  reportFatalError("__proteus_cudaGetSymbolAddress_ptr is not initialized. "
                   "Ensure the CUDA runtime is properly linked.");
}

inline cudaError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                      dim3 BlockDim, void **KernelArgs,
                                      uint64_t ShmemSize, CUstream Stream) {
  if (__proteus_cudaLaunchKernel_ptr) {
    return __proteus_cudaLaunchKernel_ptr(KernelFunc, GridDim, BlockDim,
                                          KernelArgs, ShmemSize, Stream);
  }

  reportFatalError("__proteus_cudaLaunchKernel_ptr is not initialized. Ensure "
                   "the CUDA runtime is properly linked.");
}

inline CUfunction getKernelFunctionFromImage(
    StringRef KernelName, const void *Image, bool RelinkGlobalsByCopy,
    const std::unordered_map<std::string, GlobalVarInfo> &VarNameToGlobalInfo) {
  CUfunction KernelFunc;
  CUmodule Mod;

  proteusCuErrCheck(cuModuleLoadData(&Mod, Image));
  if (RelinkGlobalsByCopy) {
    for (auto &[GlobalName, GVI] : VarNameToGlobalInfo) {
      if (!GVI.DevAddr)
        reportFatalError("Cannot copy to Global Var " + GlobalName +
                         " without a concrete device address");

      CUdeviceptr Dptr;
      size_t Bytes;
      proteusCuErrCheck(
          cuModuleGetGlobal(&Dptr, &Bytes, Mod, (GlobalName + "$ptr").c_str()));

      uint64_t PtrVal = (uint64_t)GVI.DevAddr;
      proteusCuErrCheck(cuMemcpyHtoD(Dptr, &PtrVal, Bytes));
    }
  }
  proteusCuErrCheck(
      cuModuleGetFunction(&KernelFunc, Mod, KernelName.str().c_str()));

  return KernelFunc;
}

inline cudaError_t launchKernelFunction(CUfunction KernelFunc, dim3 GridDim,
                                        dim3 BlockDim, void **KernelArgs,
                                        uint64_t ShmemSize, CUstream Stream) {
  // Convert CUresult to cudaError_t for the caller, where we replace a
  // cudaLaunchKernel call with cuLaunchKernel for the JIT module.
  auto CUresultToCudaError = [](CUresult Res) -> cudaError_t {
    switch (Res) {
    case CUDA_SUCCESS:
      return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
      return cudaErrorInvalidValue;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return cudaErrorLaunchTimeout;
    case CUDA_ERROR_LAUNCH_FAILED:
      return cudaErrorLaunchFailure;
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return cudaErrorSharedObjectInitFailed;
    case CUDA_ERROR_INVALID_HANDLE:
      return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_READY:
      return cudaErrorNotReady;
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return cudaErrorIllegalAddress;
    default:
      return cudaErrorUnknown;
    }
  };

  CUresult Res = cuLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                                BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                                Stream, KernelArgs, nullptr);
  return static_cast<cudaError_t>(CUresultToCudaError(Res));
}

} // namespace proteus

#endif
