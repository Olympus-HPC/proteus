// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_mfem_style_launch.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_mfem_style_launch.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "lambda_mfem_style_kernel.h"

int main() {
  auto Kernel23 = IntegralKernel<2, 3>;
  auto Kernel34 = IntegralKernel<3, 4>;
  Kernel23(1.0, 4, 5);
  gpuErrCheck(gpuDeviceSynchronize());
  Kernel34(-1.0, 6, 7);
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with double 1.000000e+00
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 3
// CHECK-COUNT-6: D1D = 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with double -1.000000e+00
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 3
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with i32 4
// CHECK-COUNT-12: Q1D = 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
// clang-format on
