// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=1 %build/lambda_mixed_captures.%ext 2>&1 | %FILECHECK %s

#include <iostream>

#include "proteus/JitInterface.h"

#include "gpu_common.h"

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  std::size_t I = blockIdx.x + threadIdx.x;
  if (I == 0)
    LB();
}

int main() {
  proteus::init();

  int A = 10;
  int B = 20;
  double C = 3.14;
  double D = 2.71;

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 2));

  auto lambda = [=,
                 A = proteus::jit_variable(A),
                 C = proteus::jit_variable(C)] __device__
      __attribute__((annotate("jit")))() {
        X[0] = A + B;
        X[1] = C + D;
      };

  proteus::register_lambda(lambda);
  kernel<<<1, 1>>>(lambda);
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 10
// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 20
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 2.71
// CHECK: x[0] = 30
// CHECK: x[1] = 5.85
