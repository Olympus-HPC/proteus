#ifndef CORE_HIP_HPP
#define CORE_HIP_HPP

#include <unordered_map>

#include <llvm/ADT/StringRef.h>

#include "proteus/UtilsHIP.h"

namespace proteus {

using namespace llvm;

inline void *resolveDeviceGlobalAddr(const void *Addr) {
  void *DevPtr = nullptr;
  proteusHipErrCheck(hipGetSymbolAddress(&DevPtr, HIP_SYMBOL(Addr)));
  assert(DevPtr && "Expected non-null device pointer for global");

  return DevPtr;
}

inline hipError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                     dim3 BlockDim, void **KernelArgs,
                                     uint64_t ShmemSize, hipStream_t Stream) {
  return hipLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

inline hipFunction_t getKernelFunctionFromImage(
    StringRef KernelName, const void *Image, bool RelinkGlobalsByCopy,
    const std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  hipModule_t HipModule;
  hipFunction_t KernelFunc;

  proteusHipErrCheck(hipModuleLoadData(&HipModule, Image));
  if (RelinkGlobalsByCopy) {
    for (auto &[GlobalName, HostAddr] : VarNameToDevPtr) {
      hipDeviceptr_t Dptr;
      size_t Bytes;
      proteusHipErrCheck(hipModuleGetGlobal(&Dptr, &Bytes, HipModule,
                                            (GlobalName + "$ptr").c_str()));

      void *DevPtr = resolveDeviceGlobalAddr(HostAddr);
      uint64_t PtrVal = (uint64_t)DevPtr;
      proteusHipErrCheck(hipMemcpyHtoD(Dptr, &PtrVal, Bytes));
    }
  }
  proteusHipErrCheck(
      hipModuleGetFunction(&KernelFunc, HipModule, KernelName.str().c_str()));

  return KernelFunc;
}

} // namespace proteus

#endif
