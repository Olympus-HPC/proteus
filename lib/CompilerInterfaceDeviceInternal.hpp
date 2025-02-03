#include "CompilerInterfaceDevice.h"

#include "CoreDevice.hpp"

// Return "auto" should resolve to cudaError_t or hipError_t.
static inline auto __jit_launch_kernel_internal(void *Kernel, dim3 GridDim,
                                                dim3 BlockDim,
                                                void **KernelArgs,
                                                uint64_t ShmemSize,
                                                void *Stream) {

  using namespace llvm;
  using namespace proteus;

  static const bool IsProteusDisabled =
      getEnvOrDefaultBool("ENV_PROTEUS_DISABLE", false);
  if (IsProteusDisabled) {
    return proteus::launchKernelDirect(
        Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
  }

  auto &Jit = JitDeviceImplT::instance();
  auto OptionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!OptionalKernelInfo) {
    return proteus::launchKernelDirect(
        Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
  }

  auto &KernelInfo = OptionalKernelInfo.value().get();

  auto PrintKernelLaunchInfo = [&]() {
    Logger::logs("proteus") << "JIT Launch Kernel\n";
    Logger::logs("proteus") << "=== Kernel Info\n";
    Logger::logs("proteus") << "KernelName " << KernelInfo.getName() << "\n";
    Logger::logs("proteus") << "Grid " << GridDim.x << ", " << GridDim.y << ", "
                            << GridDim.z << "\n";
    Logger::logs("proteus") << "Block " << BlockDim.x << ", " << BlockDim.y
                            << ", " << BlockDim.z << "\n";
    Logger::logs("proteus") << "KernelArgs " << KernelArgs << "\n";
    Logger::logs("proteus") << "ShmemSize " << ShmemSize << "\n";
    Logger::logs("proteus") << "Stream " << Stream << "\n";
    Logger::logs("proteus") << "=== End Kernel Info\n";
  };

  PROTEUS_DBG(PrintKernelLaunchInfo());

  return Jit.compileAndRun(
      KernelInfo, GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}
