// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/jit_eh | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/jit_eh | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitInterface.hpp>

// NOTE: Global variable to be modified inside JIT code must have external
// linkage to be accessible from JIT modules. Static internal linkage global
// variables could be supported by externalizing them and uniquely renaming
// them.
int GVar = 1;

__attribute__((annotate("jit"))) void modifyGVar() {
  if (GVar == 0)
    throw std::runtime_error("GVar is zero!");
  GVar += 1;
}

int main() {
  proteus::init();

  std::cout << "GVar " << GVar << std::endl;

  modifyGVar();

  std::cout << "GVar " << GVar << std::endl;

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: GVar 1
// CHECK: GVar 2
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineHost] ObjectCacheChain rank 0 with 1 level(s):
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 1 accesses 1
