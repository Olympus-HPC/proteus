// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_ptrtoint_capture | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_ptrtoint_capture | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <iostream>

#include <proteus/JitInterface.h>

template <typename F> void run(F &&Func) {
  proteus::register_lambda(Func)();
}

int main() {
  int Value = 0;
  int *Ptr = &Value;

  run([=, Bits = reinterpret_cast<int *>(proteus::jit_variable(reinterpret_cast<std::uintptr_t>(Ptr)))]
          () __attribute__((annotate("jit"))) {
            std::cout << *Bits << "\n";
            *Bits = 123;
            std::cout << *Bits << "\n";
          });

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with ptr inttoptr (i64 {{[0-9]+}} to ptr)
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with ptr inttoptr (i64 {{[0-9]+}} to ptr)
// CHECK: 0
// CHECK: 123
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// clang-format on
