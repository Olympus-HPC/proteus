// RUN: ./multi_file_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./multi_file_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include "launcher.hpp"


void foo();

int main() {
  launcher(my_kernel_body);
  gpuErrCheck(gpuDeviceSynchronize());
  foo();
  return 0;
}

// CHECK: File1 Kernel
// CHECK: File2 Kernel
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
