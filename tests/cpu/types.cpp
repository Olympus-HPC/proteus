// RUN: ./types | FileCheck %s --check-prefixes=CHECK

#include <cstdlib>

template<typename T>
__attribute__ ((annotate("jit", 1)))
void test(T arg) {
  volatile T local;
  local = arg;
}

int main(int argc, char **argv) {
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
  test((unsigned char) 'a');
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