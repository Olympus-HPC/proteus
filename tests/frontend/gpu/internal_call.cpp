// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"

#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

#include "../../gpu/gpu_common.h"

#include <iostream>

using namespace proteus;

auto createJitModule1() {
  auto J = std::make_unique<JitModule>(TARGET);

  auto KernelHandle = J->addKernel<void(double *)>("kernel");
  {
    auto &F = KernelHandle.F;
    F.beginFunction();
    {
      auto [V] = F.getArgs();

      auto X = F.defVar<double>(21);
      auto C = F.call<double(void)>("f2");
      auto Res = F.call<double(double, double)>("f3", X, C);
      V[0] = Res;

      F.ret();
    }
    F.endFunction();
  }

  {
    auto &F = J->addFunction<double(void)>("f2");
    F.beginFunction();
    {
      auto C = F.defVar<double>(2.0);
      F.ret(C);
    }
    F.endFunction();
  }

  {
    auto &F = J->addFunction<double(double, double)>("f3");
    F.beginFunction();
    {
      auto [X, C] = F.getArgs();
      auto P = F.declVar<double>();
      P = X * C;
      F.ret(P);
    }
    F.endFunction();
  }

  return std::make_tuple(std::move(J), KernelHandle);
}

int main() {
  auto [J, KernelHandle] = createJitModule1();

  double *V;
  gpuErrCheck(gpuMallocManaged(&V, sizeof(double)));

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "V " << *V << "\n";

  gpuErrCheck(gpuFree(V));

  return 0;
}

// clang-format off
// CHECK: V 42
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
