// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: ./kernel_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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
#define getThreadIdZ builtins::hip::getThreadIdZ
#define getBlockIdX builtins::hip::getBlockIdX
#define getBlockIdY builtins::hip::getBlockIdY
#define getBlockIdZ builtins::hip::getBlockIdZ
#define getBlockDimX builtins::hip::getBlockDimX
#define getBlockDimY builtins::hip::getBlockDimY
#define getBlockDimZ builtins::hip::getBlockDimZ
#define getGridDimX builtins::hip::getGridDimX
#define getGridDimY builtins::hip::getGridDimY
#define getGridDimZ builtins::hip::getGridDimZ

#elif PROTEUS_ENABLE_CUDA

#define TARGET "cuda"
#define getThreadIdX builtins::cuda::getThreadIdX
#define getThreadIdY builtins::cuda::getThreadIdY
#define getThreadIdZ builtins::cuda::getThreadIdZ
#define getBlockIdX builtins::cuda::getBlockIdX
#define getBlockIdY builtins::cuda::getBlockIdY
#define getBlockIdZ builtins::cuda::getBlockIdZ
#define getBlockDimX builtins::cuda::getBlockDimX
#define getBlockDimY builtins::cuda::getBlockDimY
#define getBlockDimZ builtins::cuda::getBlockDimZ
#define getGridDimX builtins::cuda::getGridDimX
#define getGridDimY builtins::cuda::getGridDimY
#define getGridDimZ builtins::cuda::getGridDimZ

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle =
      J.addKernel<double *, double *, double *, size_t, size_t, size_t>(
          "volume_add_3d");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &I = F.declVar<size_t>("i");
    auto &J_Var = F.declVar<size_t>("j");
    auto &K = F.declVar<size_t>("k");
    auto &A = F.getArg(0);
    auto &B = F.getArg(1);
    auto &C = F.getArg(2);
    auto &X = F.getArg(3);
    auto &Y = F.getArg(4);
    auto &Z = F.getArg(5);

    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    J_Var = F.callBuiltin(getBlockIdY) * F.callBuiltin(getBlockDimY) +
            F.callBuiltin(getThreadIdY);
    K = F.callBuiltin(getBlockIdZ) * F.callBuiltin(getBlockDimZ) +
        F.callBuiltin(getThreadIdZ);

    F.beginIf(I < X);
    {
      F.beginIf(J_Var < Y);
      {
        F.beginIf(K < Z);
        {
          auto &XY = F.declVar<size_t>("xy");
          auto &KXY = F.declVar<size_t>("kxy");
          auto &JX = F.declVar<size_t>("jx");
          auto &Idx = F.declVar<size_t>("idx");

          XY = X * Y;
          KXY = K * XY;
          JX = J_Var * X;
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

  const size_t X_SIZE = 4;
  const size_t Y_SIZE = 3;
  const size_t Z_SIZE = 2;
  const size_t total_size = X_SIZE * Y_SIZE * Z_SIZE;

  double *A, *B, *C;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * total_size));
  gpuErrCheck(gpuMallocManaged(&B, sizeof(double) * total_size));
  gpuErrCheck(gpuMallocManaged(&C, sizeof(double) * total_size));

  for (size_t i = 0; i < total_size; ++i) {
    A[i] = 1.5;
    B[i] = 2.5;
    C[i] = 0.0;
  }

  J.compile();

  constexpr unsigned BlockDimX = 2;
  constexpr unsigned BlockDimY = 2;
  constexpr unsigned BlockDimZ = 2;
  unsigned GridDimX = (X_SIZE + BlockDimX - 1) / BlockDimX;
  unsigned GridDimY = (Y_SIZE + BlockDimY - 1) / BlockDimY;
  unsigned GridDimZ = (Z_SIZE + BlockDimZ - 1) / BlockDimZ;

  std::cout << "Launching 3D kernel with Grid(" << GridDimX << ", " << GridDimY
            << ", " << GridDimZ << ") Block(" << BlockDimX << ", " << BlockDimY
            << ", " << BlockDimZ << ") for " << X_SIZE << "x" << Y_SIZE << "x"
            << Z_SIZE << " volume...\n";

  gpuErrCheck(KernelHandle.launch({GridDimX, GridDimY, GridDimZ},
                                  {BlockDimX, BlockDimY, BlockDimZ}, 0, nullptr,
                                  A, B, C, X_SIZE, Y_SIZE, Z_SIZE));

  gpuErrCheck(gpuDeviceSynchronize());

  bool Verified = true;
  for (size_t k = 0; k < Z_SIZE; ++k) {
    for (size_t j = 0; j < Y_SIZE; ++j) {
      for (size_t i = 0; i < X_SIZE; ++i) {
        size_t idx = k * (X_SIZE * Y_SIZE) + j * X_SIZE + i;
        if (C[idx] != 4.0) {
          std::cout << "Verification failed: C[" << i << "][" << j << "][" << k
                    << "] = " << C[idx] << " != 4.0 (expected)\n";
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

  return 0;
}

// clang-format off
// CHECK: 3D volume addition verification successful!
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1