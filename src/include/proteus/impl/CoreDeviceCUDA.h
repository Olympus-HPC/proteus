#ifndef PROTEUS_CORE_CUDA_H
#define PROTEUS_CORE_CUDA_H

#include "proteus/impl/GlobalVarInfo.h"
#include "proteus/impl/UtilsCUDA.h"

#include <llvm/ADT/StringRef.h>

#include <unordered_map>

namespace proteus {

inline void *resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  proteusCudaErrCheck(cudaGetSymbolAddress(&DevPtr, Addr));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
}

inline cudaError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                      dim3 BlockDim, void **KernelArgs,
                                      uint64_t ShmemSize, CUstream Stream) {
  return cudaLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                          Stream);
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
  cuLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                 BlockDim.y, BlockDim.z, ShmemSize, Stream, KernelArgs,
                 nullptr);
  return cudaGetLastError();
}

} // namespace proteus

#endif
