// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_2d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_2d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

using namespace proteus;

#if PROTEUS_ENABLE_HIP

#define TARGET "hip"
#define getThreadIdX builtins::hip::getThreadIdX
#define getThreadIdY builtins::hip::getThreadIdY
#define getBlockIdX builtins::hip::getBlockIdX
#define getBlockIdY builtins::hip::getBlockIdY
#define getBlockDimX builtins::hip::getBlockDimX
#define getBlockDimY builtins::hip::getBlockDimY
#define getGridDimX builtins::hip::getGridDimX
#define getGridDimY builtins::hip::getGridDimY

#elif PROTEUS_ENABLE_CUDA

#define TARGET "cuda"
#define getThreadIdX builtins::cuda::getThreadIdX
#define getThreadIdY builtins::cuda::getThreadIdY
#define getBlockIdX builtins::cuda::getBlockIdX
#define getBlockIdY builtins::cuda::getBlockIdY
#define getBlockDimX builtins::cuda::getBlockDimX
#define getBlockDimY builtins::cuda::getBlockDimY
#define getGridDimX builtins::cuda::getGridDimX
#define getGridDimY builtins::cuda::getGridDimY

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle = J.addKernel<double *, double *, double *, size_t, size_t>(
      "matrix_add_2d");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &Row = F.declVar<size_t>("row");
    auto &Col = F.declVar<size_t>("col");
    auto &A = F.getArg(0);
    auto &B = F.getArg(1);
    auto &C = F.getArg(2);
    auto &M = F.getArg(3);
    auto &N = F.getArg(4);

    Row = F.callBuiltin(getBlockIdY) * F.callBuiltin(getBlockDimY) +
          F.callBuiltin(getThreadIdY);
    Col = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
          F.callBuiltin(getThreadIdX);

    F.beginIf(Row < M);
    {
      F.beginIf(Col < N);
      {
        auto &Idx = F.declVar<size_t>("idx");
        Idx = Row * N + Col;
        C[Idx] = A[Idx] + B[Idx];
      }
      F.endIf();
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  const size_t M = 4;
  const size_t N = 8;
  const size_t size = M * N;

  double *A, *B, *C;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * size));
  gpuErrCheck(gpuMallocManaged(&B, sizeof(double) * size));
  gpuErrCheck(gpuMallocManaged(&C, sizeof(double) * size));

  for (size_t i = 0; i < size; ++i) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 0.0;
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
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      size_t idx = i * N + j;
      if (C[idx] != 3.0) {
        std::cout << "Verification failed: C[" << i << "][" << j
                  << "] = " << C[idx] << " != 3.0 (expected)\n";
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