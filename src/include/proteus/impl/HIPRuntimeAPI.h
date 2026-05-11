#ifndef PROTEUS_HIP_RUNTIME_API_H
#define PROTEUS_HIP_RUNTIME_API_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

namespace proteus::hipdyn {

const char *getErrorString(hipError_t Error);
hipError_t getDeviceProperties(hipDeviceProp_t *Prop, int DeviceId);
hipError_t getSymbolAddress(void **DevPtr, const void *Symbol);
hipError_t memcpyHtoD(hipDeviceptr_t Dst, const void *Src, size_t SizeBytes);
hipError_t moduleLoadData(hipModule_t *Module, const void *Image);
hipError_t moduleGetGlobal(hipDeviceptr_t *Dptr, size_t *Bytes,
                           hipModule_t Module, const char *Name);
hipError_t moduleGetFunction(hipFunction_t *Function, hipModule_t Module,
                             const char *Name);
hipError_t moduleLaunchKernel(hipFunction_t Function, unsigned int GridDimX,
                              unsigned int GridDimY, unsigned int GridDimZ,
                              unsigned int BlockDimX, unsigned int BlockDimY,
                              unsigned int BlockDimZ,
                              unsigned int SharedMemBytes, hipStream_t Stream,
                              void **KernelParams, void **Extra);
hipError_t launchKernel(const void *FunctionAddress, dim3 NumBlocks,
                        dim3 DimBlocks, void **Args, size_t SharedMemBytes,
                        hipStream_t Stream);
hipError_t funcSetAttribute(const void *Function, hipFuncAttribute Attribute,
                            int Value);

const char *getRTCErrorString(hiprtcResult Result);
hiprtcResult rtcLinkCreate(unsigned int NumOptions, hiprtcJIT_option *Options,
                           void **OptionValues, hiprtcLinkState *LinkStateOut);
hiprtcResult rtcLinkAddData(hiprtcLinkState LinkState,
                            hiprtcJITInputType InputType, void *Image,
                            size_t ImageSize, const char *Name,
                            unsigned int NumOptions, hiprtcJIT_option *Options,
                            void **OptionValues);
hiprtcResult rtcLinkComplete(hiprtcLinkState LinkState, void **BinOut,
                             size_t *SizeOut);

} // namespace proteus::hipdyn

#endif
