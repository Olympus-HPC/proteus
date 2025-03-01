// clang-format off
// RUN: rm -rf .proteus
// RUN: ./multi_file_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./multi_file_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include "launcher.hpp"
#include <proteus/JitInterface.hpp>

void foo();

int main() {
  proteus::init();

  gpuErrCheck(launcher(kernel_body));
  gpuErrCheck(gpuDeviceSynchronize());
  foo();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel body
// CHECK: Kernel body
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
