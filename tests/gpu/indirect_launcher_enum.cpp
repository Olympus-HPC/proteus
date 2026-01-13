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

__global__ __attribute__((annotate("jit", 1))) void print_color(color arg) {
  switch (arg){
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
  return;
}

__global__ __attribute__((annotate("jit", 1))) void print_compiler(compiler arg) {
switch (arg){
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
}

template <typename T, typename Arg> gpuError_t launcher(T KernelIn, Arg A) {
  void *Args[] = {&A};
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, Args, 0, 0);
}

int main() {
  proteus::init();
  color Color = color::Green;
  compiler Compiler = compiler::Clang;

  gpuErrCheck(gpuDeviceSynchronize());
  gpuErrCheck(launcher(print_color, Color));
  gpuErrCheck(gpuDeviceSynchronize());
  gpuErrCheck(launcher(print_compiler, Compiler));
  gpuErrCheck(gpuDeviceSynchronize());
  
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
