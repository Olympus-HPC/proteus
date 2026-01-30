// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=1 %build/lambda_pointer_captures.%ext 2>&1 | %FILECHECK %s

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

  int Scalar = 42;
  int *Ptr = &Scalar;
  double Value = 3.14;

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 2));

  auto lambda = [=] __device__ __attribute__((annotate("jit"))) () {
    X[0] = Scalar;
    X[1] = Value;
    (void)Ptr;
  };

  int *PtrOnly = &Scalar;
  auto lambda2 = [=] __device__ __attribute__((annotate("jit"))) () {
    (void)PtrOnly;
  };

  proteus::register_lambda(lambda);
  proteus::register_lambda(lambda2);

  kernel<<<1, 1>>>(lambda);
  kernel<<<1, 1>>>(lambda2);
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-NOT: [LambdaSpec][Auto]{{.*}}Ptr
// CHECK-NOT: [LambdaSpec][Auto]{{.*}}PtrOnly
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
