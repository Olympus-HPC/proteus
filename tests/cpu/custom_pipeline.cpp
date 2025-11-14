// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O3>' %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK,CHECK3
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O2>' %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK,CHECK2
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O1>' %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK,CHECK1
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Os>' %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK,CHECKs
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Oz>' %build/custom_pipeline | %FILECHECK %s --check-prefixes=CHECK,CHECKz
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/JitInterface.hpp>

__attribute__((annotate("jit"))) void foo() { std::cout << "foo" << "\n"; }

int main() {
  proteus::init();
  foo();
  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK1: [CustomPipeline] default<O1>
// CHECK2: [CustomPipeline] default<O2>
// CHECK3: [CustomPipeline] default<O3>
// CHECKs: [CustomPipeline] default<Os>
// CHECKz: [CustomPipeline] default<Oz>
// CHECK: foo
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
