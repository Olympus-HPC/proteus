// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./types_jit_array.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./types_jit_array.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

// The size of array must be provided at compile time.
#define N 3

template <typename T> __global__ void kernel(T *A) {
  proteus::jit_array(A, N);

  printf("A: ");
  for (size_t I = 0; I < N; ++I) {
    if constexpr (std::is_same_v<T, int>)
      printf("[%lu] = %d, ", I, A[I]);
    else if constexpr (std::is_same_v<T, long>)
      printf("[%lu] = %ld, ", I, A[I]);
    else if constexpr (std::is_same_v<T, unsigned>)
      printf("[%lu] = %u, ", I, A[I]);
    else if constexpr (std::is_same_v<T, unsigned long>)
      printf("[%lu] = %lu, ", I, A[I]);
    else if constexpr (std::is_same_v<T, long long>)
      printf("[%lu] = %lld, ", I, A[I]);
    else if constexpr (std::is_same_v<T, unsigned long long>)
      printf("[%lu] = %llu, ", I, A[I]);
    else if constexpr (std::is_same_v<T, float>)
      printf("[%lu] = %f, ", I, A[I]);
    else if constexpr (std::is_same_v<T, double>)
      printf("[%lu] = %lf, ", I, A[I]);
    else if constexpr (std::is_same_v<T, long double>)
      printf("[%lu] = %lf, ", I, (double)A[I]);
    else if constexpr (std::is_same_v<T, char>)
      printf("[%lu] = %d, ", I, (int)A[I]);
    else if constexpr (std::is_same_v<T, unsigned char>)
      printf("[%lu] = %d, ", I, (int)A[I]);
    else {
      printf("Unsupported type\n");
      break;
    }
  }
  printf("\n");
}

template <typename T> void launcher() {
  T *A;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(T) * N));
  for (size_t I = 0; I < N; ++I)
    if constexpr (std::is_pointer_v<T>)
      A[I] = (int *)(N - I);
    else
      A[I] = (N - I);

  kernel<<<1, 1>>>(A);
  gpuErrCheck(gpuDeviceSynchronize());

  gpuErrCheck(gpuFree(A));
}

int main() {
  proteus::init();

  launcher<int>();
  launcher<long>();
  launcher<unsigned>();
  launcher<unsigned long>();
  launcher<long long>();
  launcher<unsigned long long>();
  launcher<float>();
  launcher<double>();
  // TODO: long double demotes to double for GPUs, support if there is a
  // compelling use case. launcher<long double>();
  launcher<char>();
  launcher<unsigned char>();
  // TODO: support array of pointers if there is a compelling use case.
  // launcher<int *>();

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIiEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i32] [i32 3, i32 2, i32 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIlEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i64] [i64 3, i64 2, i64 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIjEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i32] [i32 3, i32 2, i32 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelImEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i64] [i64 3, i64 2, i64 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIxEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i64] [i64 3, i64 2, i64 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIyEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i64] [i64 3, i64 2, i64 1]
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIfEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x float] [float 3.000000e+00, float 2.000000e+00, float 1.000000e+00]
// CHECK: A: [0] = 3.000000, [1] = 2.000000, [2] = 1.000000,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIdEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x double] [double 3.000000e+00, double 2.000000e+00, double 1.000000e+00]
// CHECK: A: [0] = 3.000000, [1] = 2.000000, [2] = 1.000000,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIcEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i8] c"\03\02\01"
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIhEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [3 x i8] c"\03\02\01"
// CHECK: A: [0] = 3, [1] = 2, [2] = 1,
// CHECK: JitCache hits 0 total 10
// CHECK-COUNT-10: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 10
// CHECK-SECOND: JitStorageCache hits 10 total 10
