#include "proteus/Error.h"

#include <cuda.h>

#include <dlfcn.h>

#include <mutex>
#include <string>

namespace {

void *getCUDADriverHandle() {
  static void *Handle = nullptr;
  static std::once_flag Once;
  std::call_once(Once, []() {
    dlerror();
    Handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!Handle) {
      const char *Err = dlerror();
      std::string Message =
          "CUDA functionality requires the NVIDIA driver library libcuda.so.1, "
          "but Proteus could not load it";
      Message += Err ? std::string(": ") + Err : ".";
      proteus::reportFatalError(Message);
    }
  });
  return Handle;
}

template <typename Fn> Fn resolveCUDADriverSymbol(const char *Name) {
  void *Handle = getCUDADriverHandle();
  dlerror();
  void *Symbol = dlsym(Handle, Name);
  if (const char *Err = dlerror()) {
    proteus::reportFatalError("Failed to resolve CUDA driver symbol " +
                              std::string(Name) + ": " + Err);
  }
  return reinterpret_cast<Fn>(Symbol);
}

} // namespace

extern "C" {

CUresult CUDAAPI cuGetErrorString(CUresult Error, const char **PStr) {
  using Fn = decltype(&cuGetErrorString);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuGetErrorString");
  return Func(Error, PStr);
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  using Fn = decltype(&cuInit);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuInit");
  return Func(Flags);
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *Pctx) {
  using Fn = decltype(&cuCtxGetCurrent);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuCtxGetCurrent");
  return Func(Pctx);
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext Ctx) {
  using Fn = decltype(&cuCtxSetCurrent);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuCtxSetCurrent");
  return Func(Ctx);
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *Device) {
  using Fn = decltype(&cuCtxGetDevice);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuCtxGetDevice");
  return Func(Device);
}

CUresult CUDAAPI cuDeviceGet(CUdevice *Device, int Ordinal) {
  using Fn = decltype(&cuDeviceGet);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuDeviceGet");
  return Func(Device, Ordinal);
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *Pctx, CUdevice Dev) {
  using Fn = decltype(&cuDevicePrimaryCtxRetain);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuDevicePrimaryCtxRetain");
  return Func(Pctx, Dev);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *Pi, CUdevice_attribute Attrib,
                                      CUdevice Dev) {
  using Fn = decltype(&cuDeviceGetAttribute);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuDeviceGetAttribute");
  return Func(Pi, Attrib, Dev);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *Module, const void *Image) {
  using Fn = decltype(&cuModuleLoadData);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuModuleLoadData");
  return Func(Module, Image);
}

CUresult CUDAAPI cuModuleUnload(CUmodule Hmod) {
  using Fn = decltype(&cuModuleUnload);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuModuleUnload");
  return Func(Hmod);
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *Hfunc, CUmodule Hmod,
                                     const char *Name) {
  using Fn = decltype(&cuModuleGetFunction);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuModuleGetFunction");
  return Func(Hfunc, Hmod, Name);
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *Dptr, size_t *Bytes,
                                   CUmodule Hmod, const char *Name) {
  using Fn = decltype(&cuModuleGetGlobal);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuModuleGetGlobal");
  return Func(Dptr, Bytes, Hmod, Name);
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr DstDevice, const void *SrcHost,
                              size_t ByteCount) {
  using Fn = decltype(&cuMemcpyHtoD);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuMemcpyHtoD");
  return Func(DstDevice, SrcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH(void *DstHost, CUdeviceptr SrcDevice,
                              size_t ByteCount) {
  using Fn = decltype(&cuMemcpyDtoH);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuMemcpyDtoH");
  return Func(DstHost, SrcDevice, ByteCount);
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction Hfunc,
                                    CUfunction_attribute Attrib, int Value) {
  using Fn = decltype(&cuFuncSetAttribute);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuFuncSetAttribute");
  return Func(Hfunc, Attrib, Value);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction F, unsigned int GridDimX,
                                unsigned int GridDimY, unsigned int GridDimZ,
                                unsigned int BlockDimX, unsigned int BlockDimY,
                                unsigned int BlockDimZ,
                                unsigned int SharedMemBytes, CUstream HStream,
                                void **KernelParams, void **Extra) {
  using Fn = decltype(&cuLaunchKernel);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuLaunchKernel");
  return Func(F, GridDimX, GridDimY, GridDimZ, BlockDimX, BlockDimY, BlockDimZ,
              SharedMemBytes, HStream, KernelParams, Extra);
}

CUresult CUDAAPI cuLinkCreate(unsigned int NumOptions, CUjit_option *Options,
                              void **OptionValues, CUlinkState *StateOut) {
  using Fn = decltype(&cuLinkCreate);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuLinkCreate");
  return Func(NumOptions, Options, OptionValues, StateOut);
}

CUresult CUDAAPI cuLinkAddData(CUlinkState State, CUjitInputType Type,
                               void *Data, size_t Size, const char *Name,
                               unsigned int NumOptions, CUjit_option *Options,
                               void **OptionValues) {
  using Fn = decltype(&cuLinkAddData);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuLinkAddData");
  return Func(State, Type, Data, Size, Name, NumOptions, Options, OptionValues);
}

CUresult CUDAAPI cuLinkComplete(CUlinkState State, void **CubinOut,
                                size_t *SizeOut) {
  using Fn = decltype(&cuLinkComplete);
  static Fn Func = resolveCUDADriverSymbol<Fn>("cuLinkComplete");
  return Func(State, CubinOut, SizeOut);
}

} // extern "C"
