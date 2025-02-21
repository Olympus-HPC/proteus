#ifndef PROTEUS_CORE_CUDA_HPP
#define PROTEUS_CORE_CUDA_HPP

#include <unordered_map>

#include <llvm/ADT/StringRef.h>

#include "proteus/UtilsCUDA.h"

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
    const std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  CUfunction KernelFunc;
  CUmodule Mod;

  proteusCuErrCheck(cuModuleLoadData(&Mod, Image));
  if (RelinkGlobalsByCopy) {
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

} // namespace proteus

#endif
