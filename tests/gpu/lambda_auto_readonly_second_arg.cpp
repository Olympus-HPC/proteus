// clang-format off
// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=specialization %build/lambda_auto_readonly_second_arg.%ext 2>&1 | %FILECHECK %s
// clang-format on

#include <cstddef>
#include <iostream>

#include "proteus/JitInterface.h"

#include "gpu_common.h"

struct PrefixArgs {
  double *PadPtr;
  int A;
  double B;
  float C;
  bool D;
};

static_assert(offsetof(PrefixArgs, A) == 8);
static_assert(offsetof(PrefixArgs, B) == 16);
static_assert(offsetof(PrefixArgs, C) == 24);
static_assert(offsetof(PrefixArgs, D) == 28);

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(PrefixArgs Prefix,
                                                        T LB) {
  std::size_t I = blockIdx.x + threadIdx.x;
  if (I == 0)
    LB();
  if (Prefix.PadPtr == nullptr)
    return;
}

int main() {
  int A = 42;
  double B = 3.14;
  float C = 2.5f;
  bool D = true;

  PrefixArgs Prefix{
      /*PadPtr=*/nullptr,
      /*A=*/7,
      /*B=*/1.25,
      /*C=*/9.5f,
      /*D=*/false,
  };

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 4));

  auto lambda = [=] __device__ __attribute__((annotate("jit"))) () {
    X[0] = A;
    X[1] = B;
    X[2] = C;
    X[3] = D ? 1.0 : 0.0;
  };

  proteus::register_lambda(lambda);
  kernel<<<1, 1>>>(Prefix, lambda);
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";
  std::cout << "x[2] = " << X[2] << "\n";
  std::cout << "x[3] = " << X[3] << "\n";

  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with float 2.5
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i8 1
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
// CHECK: x[2] = 2.5
// CHECK: x[3] = 1
