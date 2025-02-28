// RUN: rm -rf .proteus
// RUN: ./types | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstdlib>

#include <proteus/JitInterface.hpp>

template <typename T> __attribute__((annotate("jit", 1))) void test(T Arg) {
  volatile T Local;
  Local = Arg;
}

int main(int argc, char **argv) {
  proteus::init();

  test(1);
  test(1l);
  test(1u);
  test(1ul);
  test(1ll);
  test(1ull);
  test(1.0f);
  test(1.0);
  test(1.0l);
  test(true);
  test('a');
  test((unsigned char)'a');

  proteus::finalize();
  return 0;
}

// CHECK: JitCache hits 0 total 12
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
