// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_unroll.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_unroll.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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

int main() {
  proteus::init();

  auto J = proteus::JitModule(TARGET);
  auto KernelHandle = J.addKernel<void(double *, int)>("for_unroll");
  auto &F = KernelHandle.F;

  auto I = F.declVar<int>("i");
  auto Inc = F.declVar<int>("inc");
  auto &Arr = F.getArg<0>();
  auto &N = F.getArg<1>();

  F.beginFunction();
  {
    Inc = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;

    F.forLoop({I, Zero, N, Inc}, [&]() { Arr[I] = Arr[I] + 1.0; })
        .unroll(4)
        .emit();

    F.ret();
  }
  F.endFunction();

  std::cout << "=== For Unroll IR ===\n" << std::flush;
  J.print();
  J.compile();

  constexpr int NElems = 16;
  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * NElems));
  for (int I = 0; I < NElems; I++) {
    X[I] = 1.0;
  }

  gpuErrCheck(
      KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X, NElems));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "For Unroll Results:\n";
  for (int I = 0; I < NElems; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  gpuErrCheck(gpuFree(X));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: === For Unroll IR ===
// CHECK: br {{.*}}!llvm.loop [[LOOP:![0-9]+]]
// CHECK: [[LOOP]] = distinct !{[[LOOP]], [[UNROLL_ENABLE:![0-9]+]], [[UNROLL_COUNT:![0-9]+]]}
// CHECK: [[UNROLL_ENABLE]] = !{!"llvm.loop.unroll.enable"}
// CHECK: [[UNROLL_COUNT]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: For Unroll Results:
// CHECK-NEXT: X[0] = 2
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 2
// CHECK-NEXT: X[3] = 2
// CHECK-NEXT: X[4] = 2
// CHECK-NEXT: X[5] = 2
// CHECK-NEXT: X[6] = 2
// CHECK-NEXT: X[7] = 2
// CHECK-NEXT: X[8] = 2
// CHECK-NEXT: X[9] = 2
// CHECK-NEXT: X[10] = 2
// CHECK-NEXT: X[11] = 2
// CHECK-NEXT: X[12] = 2
// CHECK-NEXT: X[13] = 2
// CHECK-NEXT: X[14] = 2
// CHECK-NEXT: X[15] = 2
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
