// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_2d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_2d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

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
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle =
      J.addKernelTT<void(double *, double *, double *, size_t, size_t)>(
          "matrix_add_2d");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto Row = F.declVarTT<size_t>("row");
    auto Col = F.declVarTT<size_t>("col");
    auto &A = F.getArgTT<0>();
    auto &B = F.getArgTT<1>();
    auto &C = F.getArgTT<2>();
    auto &M = F.getArgTT<3>();
    auto &N = F.getArgTT<4>();

    Row = F.callBuiltin(getBlockIdY) * F.callBuiltin(getBlockDimY) +
          F.callBuiltin(getThreadIdY);
    Col = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
          F.callBuiltin(getThreadIdX);

    F.beginIfTT(Row < M);
    {
      F.beginIfTT(Col < N);
      {
        auto Idx = F.declVarTT<size_t>("idx");
        Idx = Row * N + Col;
        C[Idx] = A[Idx] + B[Idx];
      }
      F.endIfTT();
    }
    F.endIfTT();

    F.ret();
  }
  F.endFunction();

  const size_t M = 4;
  const size_t N = 8;
  const size_t Size = M * N;

  double *A, *B, *C;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * Size));
  gpuErrCheck(gpuMallocManaged(&B, sizeof(double) * Size));
  gpuErrCheck(gpuMallocManaged(&C, sizeof(double) * Size));

  for (size_t I = 0; I < Size; ++I) {
    A[I] = 1.0;
    B[I] = 2.0;
    C[I] = 0.0;
  }

  J.compile();

  constexpr unsigned BlockDimX = 4;
  constexpr unsigned BlockDimY = 2;
  unsigned GridDimX = (N + BlockDimX - 1) / BlockDimX;
  unsigned GridDimY = (M + BlockDimY - 1) / BlockDimY;

  std::cout << "Launching 2D kernel with Grid(" << GridDimX << ", " << GridDimY
            << ") Block(" << BlockDimX << ", " << BlockDimY << ") for " << M
            << "x" << N << " matrix...\n";

  gpuErrCheck(KernelHandle.launch({GridDimX, GridDimY, 1},
                                  {BlockDimX, BlockDimY, 1}, 0, nullptr, A, B,
                                  C, M, N));

  gpuErrCheck(gpuDeviceSynchronize());

  bool Verified = true;
  for (size_t I = 0; I < M; ++I) {
    for (size_t J = 0; J < N; ++J) {
      size_t Idx = I * N + J;
      if (C[Idx] != 3.0) {
        std::cout << "Verification failed: C[" << I << "][" << J
                  << "] = " << C[Idx] << " != 3.0 (expected)\n";
        Verified = false;
      }
    }
  }

  if (Verified)
    std::cout << "2D matrix addition verification successful!\n";

  gpuErrCheck(gpuFree(A));
  gpuErrCheck(gpuFree(B));
  gpuErrCheck(gpuFree(C));

  return 0;
}

// clang-format off
// CHECK: 2D matrix addition verification successful!
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
