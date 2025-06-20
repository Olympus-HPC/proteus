// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./types_api | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// clang-format on

#include <cstdlib>

#include <iostream>

#include <proteus/JitInterface.hpp>

template <typename T> void test(T Arg) {
  proteus::jit_arg(Arg);
  volatile T Local;
  Local = Arg;
  std::cout << "Arg " << Arg << "\n";
}

int main() {
  proteus::init();

  test(1);
  test(2l);
  test(3u);
  test(4ul);
  test(5ll);
  test(6ull);
  test(7.0f);
  test(8.0);
  test(9.0l);
  test(true);
  test('a');
  test((unsigned char)'a');
  int *Ptr = (int *)0x123;
  test(Ptr);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: [ArgSpec] Replaced Function _Z4testIiEvT_ ArgNo 0 with value i32 1
// CHECK: Arg 1
// CHECK: [ArgSpec] Replaced Function _Z4testIlEvT_ ArgNo 0 with value i64 2
// CHECK: Arg 2
// CHECK: [ArgSpec] Replaced Function _Z4testIjEvT_ ArgNo 0 with value i32 3
// CHECK: Arg 3
// CHECK: [ArgSpec] Replaced Function _Z4testImEvT_ ArgNo 0 with value i64 4
// CHECK: Arg 4
// CHECK: [ArgSpec] Replaced Function _Z4testIxEvT_ ArgNo 0 with value i64 5
// CHECK: Arg 5
// CHECK: [ArgSpec] Replaced Function _Z4testIyEvT_ ArgNo 0 with value i64 6
// CHECK: Arg 6
// CHECK: [ArgSpec] Replaced Function _Z4testIfEvT_ ArgNo 0 with value float 7.000000e+00
// CHECK: Arg 7
// CHECK: [ArgSpec] Replaced Function _Z4testIdEvT_ ArgNo 0 with value double 8.000000e+00
// CHECK: Arg 8
// CHECK: [ArgSpec] Replaced Function _Z4testI{{e|g}}EvT_ ArgNo 0 with value {{x86_fp80 0xK40029000000000000000|ppc_fp128 0xM40220000000000000000000000000000}}
// CHECK: Arg 9
// CHECK: [ArgSpec] Replaced Function _Z4testIbEvT_ ArgNo 0 with value i1 true
// CHECK: Arg 1
// CHECK: [ArgSpec] Replaced Function _Z4testIcEvT_ ArgNo 0 with value i8 97
// CHECK: Arg a
// CHECK: [ArgSpec] Replaced Function _Z4testIhEvT_ ArgNo 0 with value i8 97
// CHECK: Arg a
// CHECK: [ArgSpec] Replaced Function _Z4testIPiEvT_ ArgNo 0 with value ptr inttoptr (i64 291 to ptr)
// CHECK: Arg 0x123
// CHECK: JitCache hits 0 total 13
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
