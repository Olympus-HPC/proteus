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
#include <proteus/JitInterface.h>

enum color { Red = 1, Yellow = 0, Green = -2 };

enum class compiler { Clang, GCC, NVCC };

enum class Small : uint8_t { 
  A = 255,
  B = 42,
};

enum class Flags : uint32_t {
  None = 0,
  HighBit = 0x80000000u
};

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

__device__ void nativeEcho(Small v) {
  // Should always truncate and treat as uint8_t
  printf("VALUE %hhu\n", static_cast<uint8_t>(v));
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
  Small a = Small::A;
  Small b = Small::B;
  Flags f = Flags::HighBit;
  bool native_negative =
      static_cast<uint32_t>(f) < 0;  // always false

  auto color_lambda = [ =, Color = proteus::jit_variable(Color) ] __device__()
      __attribute__((annotate("jit"))) {
    switch (Color) {
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
  auto compiler_lambda =
      [ =, Compiler = proteus::jit_variable(Compiler) ] __device__()
          __attribute__((annotate("jit"))) {
    switch (Compiler) {
    case compiler::Clang: {
      printf("clang\n");
      break;
    }
    case compiler::GCC: {
      printf("gcc\n");
      break;
    }
    case compiler::NVCC: {
      printf("nvcc\n");
      break;
    }
    default:
      break;
    }
    return;
  };

  auto uint_lambda = [=, a = proteus::jit_variable(a), 
    b = proteus::jit_variable(b)] __attribute__((annotate("jit"))) {
    nativeEcho(a);
    if (a > b) {
      printf("Less than\n");
    } 
  };
  // // constexpr uint64_t big_int = 1 << 63;
  // auto flags_lambda = [=, f = proteus::jit_variable(f)] __attribute__((annotate("jit"))) {
  //   if (uint32_t(f) > 0) {
  //     printf("Less than\n");
  //   }
  // };
  
  run(color_lambda);
  run(compiler_lambda);
  run(uint_lambda);
  //run(flags_lambda);
  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Green
// CHECK: clang
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
