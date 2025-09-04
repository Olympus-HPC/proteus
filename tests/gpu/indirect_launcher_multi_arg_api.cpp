// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher_multi_arg_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher_multi_arg_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel(int Arg) {
  proteus::jit_arg(Arg);
  printf("Kernel one; arg = %d\n", Arg);
}

__global__ void kernelTwo(int Arg) {
  proteus::jit_arg(Arg);
  printf("Kernel two; arg = %d\n", Arg);
}

gpuError_t launcher(const void *KernelIn, int A) {
  void *Args[] = {&A};
  return gpuLaunchKernel(KernelIn, 1, 1, Args, 0, 0);
}

int main() {
  proteus::init();

  gpuErrCheck(launcher((const void *)kernel, 42));
  gpuErrCheck(launcher((const void *)kernelTwo, 24));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel one; arg = 42
// CHECK: Kernel two; arg = 24
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
