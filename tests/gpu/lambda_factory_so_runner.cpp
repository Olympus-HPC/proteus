// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory_so_runner.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_factory_so_runner.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "lambda_factory_so.h"

int main() {
  printf("runner tu1\n");
  run_lambda_factory_tu1();

  printf("runner tu2\n");
  run_lambda_factory_tu2();
  return 0;
}

// clang-format off
// CHECK: runner tu1
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 10
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 1 with i32 11
// CHECK: Integer = 10
// CHECK: Integer = 11
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 12
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 1 with i32 10
// CHECK: Integer = 12
// CHECK: Integer = 10
// CHECK: runner tu2
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 20
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 1 with i32 21
// CHECK: Integer = 20
// CHECK: Integer = 21
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 0 with i32 22
// CHECK-FIRST-DAG: [LambdaSpec] Replacing slot 1 with i32 20
// CHECK: Integer = 22
// CHECK: Integer = 20
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 4
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 4 accesses 4
// clang-format on
