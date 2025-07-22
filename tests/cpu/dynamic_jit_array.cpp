// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./dynamic_jit_array | %FILECHECK %s --check-prefixes=CHECK
// clang-format on

#include <climits>
#include <cstdio>

#include <proteus/JitInterface.hpp>

template <typename T> void testConst(T *A) {
  proteus::jit_array(A, 3);

  printf("A: ");
  for (size_t I = 0; I < 3; ++I) {
    if constexpr (std::is_same_v<T, int>)
      printf("[%lu] = %d, ", I, A[I]);
    else {
      static_assert(false, "Unsupported type\n");
    }
  }
  printf("\n");
}

template <typename T> void testRunConst(T *A, size_t N) {
  proteus::jit_arg(N);
  proteus::jit_array(A, N);

  printf("A: ");
  for (size_t I = 0; I < N; ++I) {
    if constexpr (std::is_same_v<T, int>)
      printf("[%lu] = %d, ", I, A[I]);
    else {
      static_assert(false, "Unsupported type\n");
    }
  }
  printf("\n");
}

template <typename T> void launcher() {
  size_t N = 3;
  T *A = new T[N];
  for (size_t I = 0; I < N; ++I)
    A[I] = (N - I);

  testConst(A);

  testRunConst(A, N);

  delete[] A;
}

int main() {
  proteus::init();

  launcher<int>();

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: [ArgSpec] Replaced Function _Z9testConstIiEvPT_ ArgNo 0 with value @0 = private constant [3 x i32] [i32 3, i32 2, i32 1]
// CHECK: [ArgSpec] Replaced Function _Z12testRunConstIiEvPT_m ArgNo 0 with value @0 = private constant [3 x i32] [i32 3, i32 2, i32 1]
// CHECK: [ArgSpec] Replaced Function _Z12testRunConstIiEvPT_m ArgNo 1 with value i64 3
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
