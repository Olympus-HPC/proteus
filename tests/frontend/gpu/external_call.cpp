// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/external_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/external_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

extern "C" {
__device__ void hello() { printf("Hello!\n"); }
__device__ int add(int A, int B) { return A + B; }
}

int main() {
  auto J = proteus::JitModule(TARGET);
  auto KernelHandle = J.addKernel<void(int *)>("ExternalCall");
  {
    auto &F = KernelHandle.F;
    F.beginFunction();
    {
      auto &Out = F.getArg(0);

      F.call<void(void)>("hello");
      auto &V1 = F.defVar<int>(22);
      auto &V2 = F.defVar<int>(20);
      auto &V3 = F.call<int(int, int)>("add", V1, V2);
      Out[0] = V3;

      F.ret();
    }
    F.endFunction();
  }

  int *V;
  gpuErrCheck(gpuMallocManaged(&V, sizeof(int)));

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "V " << *V << "\n";

  gpuErrCheck(gpuFree(V));

  return 0;
}

// clang-format off
// CHECK: Hello!
// CHECK-NEXT: V 42
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
