// RUN: ./types | FileCheck %s --check-prefixes=CHECK
template<typename T>
__attribute__ ((annotate("jit", 1)))
void test(T arg) {
  volatile T local;
  local = arg;
}

int main(int argc, char **argv) {
  test(1);
  test(1l);
  // test(short int{i});
  test(1.0f);
  test(1.0);
  // test(1.0l);
  //test(true);
}

// CHECK: JitCache hits 0 total 4
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0