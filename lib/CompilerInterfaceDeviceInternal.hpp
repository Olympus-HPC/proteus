#include "CompilerInterfaceDevice.h"

// Return "auto" should resolve to cudaError_t or hipError_t.
static inline auto
__jit_launch_kernel_internal(const char *ModuleUniqueId, void *Kernel,
                             dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                             uint64_t ShmemSize, void *Stream) {

  using namespace llvm;
  using namespace proteus;
  auto &Jit = JitDeviceImplT::instance();
  auto optionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!optionalKernelInfo) {
    return Jit.launchKernelDirect(
        Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
  }

  const auto &KernelInfo = optionalKernelInfo.value();
  const char *KernelName = KernelInfo.getName();
  int32_t NumRuntimeConstants = KernelInfo.getNumRCs();
  auto RCIndices = KernelInfo.getRCIndices();
  auto RCTypes = KernelInfo.getRCTypes();

  auto printKernelLaunchInfo = [&]() {
    Logger::logs("proteus") << "JIT Launch Kernel\n";
    Logger::logs("proteus") << "=== Kernel Info\n";
    Logger::logs("proteus") << "KernelName " << KernelName << "\n";
    Logger::logs("proteus") << "Grid " << GridDim.x << ", " << GridDim.y << ", "
                            << GridDim.z << "\n";
    Logger::logs("proteus") << "Block " << BlockDim.x << ", " << BlockDim.y
                            << ", " << BlockDim.z << "\n";
    Logger::logs("proteus") << "KernelArgs " << KernelArgs << "\n";
    Logger::logs("proteus") << "ShmemSize " << ShmemSize << "\n";
    Logger::logs("proteus") << "Stream " << Stream << "\n";
    Logger::logs("proteus") << "=== End Kernel Info\n";
  };

  TIMESCOPE("__jit_launch_kernel");
  DBG(printKernelLaunchInfo());

  return Jit.compileAndRun(
      ModuleUniqueId, Kernel, KernelName, RCIndices, RCTypes,
      NumRuntimeConstants, GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}
