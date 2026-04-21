// clang-format off
// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=specialization %build/lambda_field_pointer_escape.%ext 2>&1 | %FILECHECK %s
// clang-format on

#include <cstdio>
#include <iostream>

#include "gpu_common.h"
#include "proteus/JitInterface.h"

__device__ __attribute__((noinline)) void observe(const int *Ptr) {
  if (Ptr == nullptr)
    printf("null\n");
}

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  std::size_t I = blockIdx.x + threadIdx.x;
  if (I == 0)
    LB();
}

int main() {
  int A = 42;
  double B = 3.14;

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 2));

  auto lambda = [X, A, B] __device__ __attribute__((annotate("jit"))) {
    observe(&A);
    X[0] = A;
    X[1] = B;
  };

  proteus::register_lambda(lambda);
  kernel<<<1, 1>>>(lambda);
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-NOT: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
