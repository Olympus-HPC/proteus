#include "CompilerInterfaceDevice.h"

// Return "auto" should resolve to cudaError_t or hipError_t.
static inline auto
__jit_launch_kernel_internal(const char *ModuleUniqueId, void *Kernel,
                             void *FatbinWrapper, size_t FatbinSize,
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

  auto printKernelLaunchInfo = [&]() {
    dbgs() << "JIT Launch Kernel\n";
    dbgs() << "=== Kernel Info\n";
    dbgs() << "KernelName " << KernelName << "\n";
    dbgs() << "FatbinSize " << FatbinSize << "\n";
    dbgs() << "Grid " << GridDim.x << ", " << GridDim.y << ", " << GridDim.z
           << "\n";
    dbgs() << "Block " << BlockDim.x << ", " << BlockDim.y << ", " << BlockDim.z
           << "\n";
    dbgs() << "KernelArgs " << KernelArgs << "\n";
    dbgs() << "ShmemSize " << ShmemSize << "\n";
    dbgs() << "Stream " << Stream << "\n";
    dbgs() << "=== End Kernel Info\n";
  };

  TIMESCOPE("__jit_launch_kernel");
  DBG(printKernelLaunchInfo());

  return Jit.compileAndRun(
      ModuleUniqueId, KernelName,
      reinterpret_cast<FatbinWrapper_t *>(FatbinWrapper), FatbinSize, RCIndices,
      NumRuntimeConstants, GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}