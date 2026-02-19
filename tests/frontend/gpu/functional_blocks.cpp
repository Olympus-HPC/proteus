// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/functional_blocks.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/functional_blocks.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

constexpr auto Eager = proteus::EmissionPolicy::Eager;

int main() {
  auto J = proteus::JitModule(TARGET);
  auto KernelHandle = J.addKernel<void(double *, int)>("functional_blocks");
  auto &F = KernelHandle.F;

  auto I = F.declVar<int>("i");
  auto One = F.declVar<int>("one");
  auto Zero = F.declVar<int>("zero");
  auto &Arg = F.getArg<0>();
  auto &N = F.getArg<1>();

  F.function([&]() {
    One = 1;
    Zero = 0;

    I = Zero;
    F.forLoop<Eager>(I, I, N, One, [&]() {
      F.ifThen(I % 2 == 0, [&]() { Arg[I] = Arg[I] + 2.0; });
    });

    I = Zero;
    F.whileLoop([&]() { return I < N; },
                [&]() {
                  Arg[I] = Arg[I] + 1.0;
                  I = I + One;
                });

    F.ret();
  });

  J.compile();

  constexpr int NHost = 10;
  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * NHost));
  for (int I = 0; I < NHost; I++)
    X[I] = 1.0;

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X, NHost));
  gpuErrCheck(gpuDeviceSynchronize());

  for (int I = 0; I < NHost; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  gpuErrCheck(gpuFree(X));

  return 0;
}

// clang-format off
// CHECK: X[0] = 4
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 4
// CHECK-NEXT: X[3] = 2
// CHECK-NEXT: X[4] = 4
// CHECK-NEXT: X[5] = 2
// CHECK-NEXT: X[6] = 4
// CHECK-NEXT: X[7] = 2
// CHECK-NEXT: X[8] = 4
// CHECK-NEXT: X[9] = 2
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
