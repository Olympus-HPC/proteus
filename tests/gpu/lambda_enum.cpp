// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/indirect_launcher_arg.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher_arg.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

enum color {
    Red, 
    Yellow, 
    Green
};

enum class compiler {
    Clang,
    GCC,
    NVCC
};


template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

template <typename T> void run(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  proteus::init();
  color Color = color::Green;
  compiler Compiler = compiler::Clang;
  
  auto color_lambda = [=, Color = proteus::jit_variable(Color)]
                                    __device__  ()__attribute__((annotate("jit"))) {
                                        switch (Color){
                                          case Red: {
                                            printf("Red\n");
                                            break;
                                          }
                                          case Yellow: {
                                            printf("Yellow\n");
                                            break;
                                          }
                                          case Green: {
                                            printf("Green\n");
                                            break;
                                          }
                                          default:
                                          break;
                                          }
                                    };
    auto compiler_lambda = [=, Compiler = proteus::jit_variable(Compiler)]
                                    __device__  ()__attribute__((annotate("jit"))) {
                                      switch (Compiler){
                                        case compiler::Clang:{
                                          printf("clang\n");
                                          break;
                                        }
                                        case compiler::GCC:{
                                          printf("gcc\n");
                                          break;
                                        }
                                        case compiler::NVCC:{
                                          printf("nvcc\n");
                                          break;
                                        }
                                        default:
                                        break;
                                        }
                                        return;
                                    };

  run(color_lambda);
  run(compiler_lambda);
  
  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 42
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel 42
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 24
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel 24
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
