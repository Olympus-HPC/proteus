// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"

#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

#include "../../gpu/gpu_common.h"

using namespace proteus;

auto createJitModule1() {
  auto J = std::make_unique<JitModule>(TARGET);

  auto KernelHandle = J->addKernelTT<void(double *)>("kernel");
  {
    auto &F = KernelHandle.F;
    F.beginFunction();
    {
      auto [V] = F.getArgsTT();

      auto X = F.defVarTT<double>(21);
      auto C = F.callTT<double(void)>("f2");
      auto Res = F.callTT<double(double, double)>("f3", X, C);
      V[0] = Res;

      F.retTT();
    }
    F.endFunction();
  }

  {
    auto &F = J->addFunction<double(void)>("f2");
    F.beginFunction();
    {
      auto C = F.defVarTT<double>(2.0);
      F.retTT(C);
    }
    F.endFunction();
  }

  {
    auto &F = J->addFunction<double(double, double)>("f3");
    F.beginFunction();
    {
      auto [X, C] = F.getArgsTT();
      auto P = F.declVarTT<double>();
      P = X * C;
      F.retTT(P);
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
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
