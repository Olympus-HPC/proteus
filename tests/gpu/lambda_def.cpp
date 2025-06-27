// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./lambda_def.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda_def.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB(1);
}

template <typename T> void run(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  proteus::init();

  //auto Lambda = [ =, A = proteus::jit_variable(A)
  //, B = proteus::jit_variable(B) ] __device__(int)
  //    __attribute__((annotate("jit"))) {
    //  auto lam = [&]() {
      //    printf("Lambda A %d\n", A);
      //    printf("Lambda B %d\n", B);
      //  };
      //  lam();
      //};
  int A = 42;
  int B = 28;
  int C = 1;
  int D = 12;
  auto Lambda = [=,
    A = proteus::jit_variable(A),
    B = proteus::jit_variable(B),
    C = proteus::jit_variable(C),
    D = proteus::jit_variable(D)
   ]
   __device__(int)
   __attribute__((annotate("jit"))) {
    int tmp = C*A;
    auto lam = [&]() {
      auto res = proteus::shared_array<int>(C*B);
      auto res2 = proteus::shared_array<int>(D*B);
      return C*B + D*B;
    };
    return lam() + tmp;
  };
  // auto Lambda = [=, A = proteus::jit_variable(A)
  // , B = proteus::jit_variable(B)] __device__(int) {
    // double* arr = proteus::shared_array<double>(A*B);
    // arr[0] = A;
    // arr[1] = B;
    //
    // printf("Lambda A %f\n", arr[0]);
    // printf("Lambda B %f\n", arr[1]);
  // };
  run(Lambda);
  run(Lambda);
  run(Lambda);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK-COUNT-3: Lambda A 42
// CHECK: JitCache hits 2 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
