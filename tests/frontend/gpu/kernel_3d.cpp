// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

#include <iostream>

using namespace proteus;
using namespace builtins::gpu;

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

  auto KernelHandle =
      J.addKernel<void(double *, double *, double *, size_t, size_t, size_t)>(
          "volume_add_3d");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto I = F.declVar<size_t>("i");
    auto JVar = F.declVar<size_t>("j");
    auto K = F.declVar<size_t>("k");
    auto &A = F.getArg<0>();
    auto &B = F.getArg<1>();
    auto &C = F.getArg<2>();
    auto &X = F.getArg<3>();
    auto &Y = F.getArg<4>();
    auto &Z = F.getArg<5>();

    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    JVar = F.callBuiltin(getBlockIdY) * F.callBuiltin(getBlockDimY) +
           F.callBuiltin(getThreadIdY);
    K = F.callBuiltin(getBlockIdZ) * F.callBuiltin(getBlockDimZ) +
        F.callBuiltin(getThreadIdZ);

    F.beginIf(I < X);
    {
      F.beginIf(JVar < Y);
      {
        F.beginIf(K < Z);
        {
          auto XY = F.declVar<size_t>("xy");
          auto KXY = F.declVar<size_t>("kxy");
          auto JX = F.declVar<size_t>("jx");
          auto Idx = F.declVar<size_t>("idx");

          XY = X * Y;
          KXY = K * XY;
          JX = JVar * X;
          Idx = KXY + JX + I;

          C[Idx] = A[Idx] + B[Idx];
        }
        F.endIf();
      }
      F.endIf();
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  const size_t XSize = 4;
  const size_t YSize = 3;
  const size_t ZSize = 2;
  const size_t TotalSize = XSize * YSize * ZSize;

  double *A, *B, *C;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * TotalSize));
  gpuErrCheck(gpuMallocManaged(&B, sizeof(double) * TotalSize));
  gpuErrCheck(gpuMallocManaged(&C, sizeof(double) * TotalSize));

  for (size_t I = 0; I < TotalSize; ++I) {
    A[I] = 1.5;
    B[I] = 2.5;
    C[I] = 0.0;
  }

  J.compile();

  constexpr unsigned BlockDimX = 2;
  constexpr unsigned BlockDimY = 2;
  constexpr unsigned BlockDimZ = 2;
  unsigned GridDimX = (XSize + BlockDimX - 1) / BlockDimX;
  unsigned GridDimY = (YSize + BlockDimY - 1) / BlockDimY;
  unsigned GridDimZ = (ZSize + BlockDimZ - 1) / BlockDimZ;

  std::cout << "Launching 3D kernel with Grid(" << GridDimX << ", " << GridDimY
            << ", " << GridDimZ << ") Block(" << BlockDimX << ", " << BlockDimY
            << ", " << BlockDimZ << ") for " << XSize << "x" << YSize << "x"
            << ZSize << " volume...\n";

  gpuErrCheck(KernelHandle.launch({GridDimX, GridDimY, GridDimZ},
                                  {BlockDimX, BlockDimY, BlockDimZ}, 0, nullptr,
                                  A, B, C, XSize, YSize, ZSize));

  gpuErrCheck(gpuDeviceSynchronize());

  bool Verified = true;
  for (size_t K = 0; K < ZSize; ++K) {
    for (size_t J = 0; J < YSize; ++J) {
      for (size_t I = 0; I < XSize; ++I) {
        size_t Idx = K * (XSize * YSize) + J * XSize + I;
        if (C[Idx] != 4.0) {
          std::cout << "Verification failed: C[" << I << "][" << J << "][" << K
                    << "] = " << C[Idx] << " != 4.0 (expected)\n";
          Verified = false;
        }
      }
    }
  }

  if (Verified)
    std::cout << "3D volume addition verification successful!\n";

  gpuErrCheck(gpuFree(A));
  gpuErrCheck(gpuFree(B));
  gpuErrCheck(gpuFree(C));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: 3D volume addition verification successful!
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
