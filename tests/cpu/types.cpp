// RUN: rm -rf .proteus
// RUN: ./types | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstdlib>

#include <iostream>

#include <proteus/JitInterface.hpp>

template <typename T> __attribute__((annotate("jit", 1))) void test(T Arg) {
  volatile T Local;
  Local = Arg;
  std::cout << "Arg " << Arg << "\n";
}

int main(int argc, char **argv) {
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

// CHECK: Arg 1
// CHECK: Arg 2
// CHECK: Arg 3
// CHECK: Arg 4
// CHECK: Arg 5
// CHECK: Arg 6
// CHECK: Arg 7
// CHECK: Arg 8
// CHECK: Arg 9
// CHECK: Arg 1
// CHECK: Arg a
// CHECK: Arg a
// CHECK: Arg 0x123
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
