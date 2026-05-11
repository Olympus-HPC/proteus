#include "proteus/impl/HIPRuntimeAPI.h"

#include "proteus/Error.h"

#include <dlfcn.h>

#include <mutex>
#include <string>

#ifndef PROTEUS_HIP_RUNTIME_LIBRARY_SONAME
#define PROTEUS_HIP_RUNTIME_LIBRARY_SONAME "libamdhip64.so"
#endif

#ifndef PROTEUS_HIPRTC_LIBRARY_SONAME
#define PROTEUS_HIPRTC_LIBRARY_SONAME "libhiprtc.so"
#endif

#ifndef PROTEUS_HIP_INSTALL_ROOT
#define PROTEUS_HIP_INSTALL_ROOT ""
#endif

#define PROTEUS_STRINGIFY_IMPL(X) #X
#define PROTEUS_STRINGIFY(X) PROTEUS_STRINGIFY_IMPL(X)

namespace {

void *loadLibrary(const char *PrimaryName, const char *FallbackName,
                  const char *Description) {
  std::string BuildRoot = PROTEUS_HIP_INSTALL_ROOT;
  if (!BuildRoot.empty()) {
    std::string PrimaryPath = BuildRoot + "/lib/" + PrimaryName;
    dlerror();
    if (void *Handle = dlopen(PrimaryPath.c_str(), RTLD_NOW | RTLD_LOCAL))
      return Handle;

    std::string FallbackPath = BuildRoot + "/lib/" + FallbackName;
    dlerror();
    if (void *Handle = dlopen(FallbackPath.c_str(), RTLD_NOW | RTLD_LOCAL))
      return Handle;
  }

  dlerror();
  if (void *Handle = dlopen(PrimaryName, RTLD_NOW | RTLD_LOCAL))
    return Handle;
  const char *PrimaryError = dlerror();

  dlerror();
  if (void *Handle = dlopen(FallbackName, RTLD_NOW | RTLD_LOCAL))
    return Handle;
  const char *FallbackError = dlerror();

  std::string Message =
      std::string("ROCm functionality requires ") + Description +
      ", but Proteus could not load " + PrimaryName + " or " + FallbackName +
      ". Ensure a compatible system ROCm installation is available";
  if (PrimaryError || FallbackError) {
    Message += ".";
    if (PrimaryError)
      Message += " " + std::string(PrimaryName) + ": " + PrimaryError;
    if (FallbackError)
      Message += " " + std::string(FallbackName) + ": " + FallbackError;
  }
  Message += " You may need to configure LD_LIBRARY_PATH from ROCM_PATH.";
  proteus::reportFatalError(Message);
}

void *getHIPRuntimeHandle() {
  static void *Handle = nullptr;
  static std::once_flag Once;
  std::call_once(Once, []() {
    Handle = loadLibrary(PROTEUS_HIP_RUNTIME_LIBRARY_SONAME, "libamdhip64.so",
                         "the ROCm HIP runtime library");
  });
  return Handle;
}

void *getHIPRTCHandle() {
  static void *Handle = nullptr;
  static std::once_flag Once;
  std::call_once(Once, []() {
    Handle = loadLibrary(PROTEUS_HIPRTC_LIBRARY_SONAME, "libhiprtc.so",
                         "the ROCm HIPRTC library");
  });
  return Handle;
}

template <typename Fn>
Fn resolveSymbol(void *Handle, const char *Name, const char *LibraryName) {
  dlerror();
  void *Symbol = dlsym(Handle, Name);
  if (const char *Err = dlerror()) {
    proteus::reportFatalError("Failed to resolve ROCm symbol " +
                              std::string(Name) + " from " + LibraryName +
                              ": " + Err);
  }
  return reinterpret_cast<Fn>(Symbol);
}

template <typename Fn> Fn resolveHIPRuntimeSymbol(const char *Name) {
  return resolveSymbol<Fn>(getHIPRuntimeHandle(), Name, "libamdhip64");
}

template <typename Fn> Fn resolveHIPRTCSymbol(const char *Name) {
  return resolveSymbol<Fn>(getHIPRTCHandle(), Name, "libhiprtc");
}

} // namespace

namespace proteus::hipdyn {

const char *getErrorString(hipError_t Error) {
  using Fn = decltype(&::hipGetErrorString);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipGetErrorString");
  return Func(Error);
}

hipError_t getDeviceProperties(hipDeviceProp_t *Prop, int DeviceId) {
  using Fn = decltype(&::hipGetDeviceProperties);
  static Fn Func =
      resolveHIPRuntimeSymbol<Fn>(PROTEUS_STRINGIFY(hipGetDeviceProperties));
  return Func(Prop, DeviceId);
}

hipError_t getSymbolAddress(void **DevPtr, const void *Symbol) {
  using Fn = hipError_t (*)(void **, const void *);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipGetSymbolAddress");
  return Func(DevPtr, Symbol);
}

hipError_t memcpyHtoD(hipDeviceptr_t Dst, const void *Src, size_t SizeBytes) {
  using Fn = decltype(&::hipMemcpyHtoD);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipMemcpyHtoD");
  return Func(Dst, const_cast<void *>(Src), SizeBytes);
}

hipError_t moduleLoadData(hipModule_t *Module, const void *Image) {
  using Fn = decltype(&::hipModuleLoadData);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipModuleLoadData");
  return Func(Module, Image);
}

hipError_t moduleGetGlobal(hipDeviceptr_t *Dptr, size_t *Bytes,
                           hipModule_t Module, const char *Name) {
  using Fn = decltype(&::hipModuleGetGlobal);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipModuleGetGlobal");
  return Func(Dptr, Bytes, Module, Name);
}

hipError_t moduleGetFunction(hipFunction_t *Function, hipModule_t Module,
                             const char *Name) {
  using Fn = decltype(&::hipModuleGetFunction);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipModuleGetFunction");
  return Func(Function, Module, Name);
}

hipError_t moduleLaunchKernel(hipFunction_t Function, unsigned int GridDimX,
                              unsigned int GridDimY, unsigned int GridDimZ,
                              unsigned int BlockDimX, unsigned int BlockDimY,
                              unsigned int BlockDimZ,
                              unsigned int SharedMemBytes, hipStream_t Stream,
                              void **KernelParams, void **Extra) {
  using Fn = decltype(&::hipModuleLaunchKernel);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipModuleLaunchKernel");
  return Func(Function, GridDimX, GridDimY, GridDimZ, BlockDimX, BlockDimY,
              BlockDimZ, SharedMemBytes, Stream, KernelParams, Extra);
}

hipError_t launchKernel(const void *FunctionAddress, dim3 NumBlocks,
                        dim3 DimBlocks, void **Args, size_t SharedMemBytes,
                        hipStream_t Stream) {
  using Fn =
      hipError_t (*)(const void *, dim3, dim3, void **, size_t, hipStream_t);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipLaunchKernel");
  return Func(FunctionAddress, NumBlocks, DimBlocks, Args, SharedMemBytes,
              Stream);
}

hipError_t funcSetAttribute(const void *Function, hipFuncAttribute Attribute,
                            int Value) {
  using Fn = hipError_t (*)(const void *, hipFuncAttribute, int);
  static Fn Func = resolveHIPRuntimeSymbol<Fn>("hipFuncSetAttribute");
  return Func(Function, Attribute, Value);
}

const char *getRTCErrorString(hiprtcResult Result) {
  using Fn = decltype(&::hiprtcGetErrorString);
  static Fn Func = resolveHIPRTCSymbol<Fn>("hiprtcGetErrorString");
  return Func(Result);
}

hiprtcResult rtcLinkCreate(unsigned int NumOptions, hiprtcJIT_option *Options,
                           void **OptionValues, hiprtcLinkState *LinkStateOut) {
  using Fn = decltype(&::hiprtcLinkCreate);
  static Fn Func = resolveHIPRTCSymbol<Fn>("hiprtcLinkCreate");
  return Func(NumOptions, Options, OptionValues, LinkStateOut);
}

hiprtcResult rtcLinkAddData(hiprtcLinkState LinkState,
                            hiprtcJITInputType InputType, void *Image,
                            size_t ImageSize, const char *Name,
                            unsigned int NumOptions, hiprtcJIT_option *Options,
                            void **OptionValues) {
  using Fn = decltype(&::hiprtcLinkAddData);
  static Fn Func = resolveHIPRTCSymbol<Fn>("hiprtcLinkAddData");
  return Func(LinkState, InputType, Image, ImageSize, Name, NumOptions, Options,
              OptionValues);
}

hiprtcResult rtcLinkComplete(hiprtcLinkState LinkState, void **BinOut,
                             size_t *SizeOut) {
  using Fn = decltype(&::hiprtcLinkComplete);
  static Fn Func = resolveHIPRTCSymbol<Fn>("hiprtcLinkComplete");
  return Func(LinkState, BinOut, SizeOut);
}

} // namespace proteus::hipdyn
