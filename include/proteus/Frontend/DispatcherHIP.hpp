#ifndef PROTEUS_FRONTEND_DISPATCHER_HIP_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HIP_HPP

#if PROTEUS_ENABLE_HIP

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/JitEngineDeviceHIP.hpp"

namespace proteus {

class DispatcherHIP : public Dispatcher {
public:
  static DispatcherHIP &instance() {
    static DispatcherHIP D;
    return D;
  }

  void compile(std::unique_ptr<Module> M) override {
    Library = Jit.compileOnly(*M);
    if (!Library)
      PROTEUS_FATAL_ERROR("Expected non-null object library");

    {
      std::error_code EC;
      raw_fd_ostream OS{"object.o", EC};
      OS << Library->getBuffer();
    }
  }

  DispatchResult launch(StringRef KernelName, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream) override {
    auto KernelFunc = proteus::getKernelFunctionFromImage(
        KernelName, Library->getBufferStart(),
        /*RelinkGlobalsByCopy*/ false,
        /* VarNameToDevPtr */ {});

    dim3 HipGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 HipBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    hipStream_t HipStream = reinterpret_cast<hipStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(KernelFunc, HipGridDim, HipBlockDim,
                                         KernelArgsPtrs, ShmemSize, HipStream);
  }

protected:
  void *getFunctionAddress(StringRef) override {
    PROTEUS_FATAL_ERROR("HIP does not support getFunctionAddress");
  }

private:
  JitEngineDeviceHIP &Jit;
  DispatcherHIP() : Jit(JitEngineDeviceHIP::instance()) {}
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_HPP
