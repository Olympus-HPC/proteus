// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/convert.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/convert.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  proteus::init();

  auto J = proteus::JitModule(TARGET);
  auto KernelHandle =
      J.addKernel<void(double *, int *, float *, double *, int *, float *)>(
          "cast");
  auto &F = KernelHandle.F;
  auto &DOut = F.getArg(0);
  auto &IOut = F.getArg(1);
  auto &FOut = F.getArg(2);
  auto &DOut2 = F.getArg(3);
  auto &IOut2 = F.getArg(4);
  auto &FOut2 = F.getArg(5);

  F.beginFunction();
  {
    auto &Di = F.declVar<double>();
    auto &Ii = F.declVar<int>();
    auto &Fi = F.declVar<float>();

    Di = 3.9;
    Ii = -7;
    Fi = 2.5f;

    auto &IfromD = F.convert<int>(Di);
    auto &FfromI = F.convert<float>(Ii);
    auto &DfromF = F.convert<double>(Fi);

    DOut[0] = DfromF;
    IOut[0] = IfromD;
    FOut[0] = FfromI;

    auto &D2 = F.convert<double>(IfromD);
    auto &I2 = F.convert<int>(FfromI);
    auto &F2 = F.convert<float>(DfromF);

    DOut2[0] = D2;
    IOut2[0] = I2;
    FOut2[0] = F2;

    F.ret();
  }
  F.endFunction();

  J.compile();

  double *OutDoubleFromFloat, *OutDoubleFromInt;
  int *OutIntFromDouble, *OutIntFromFloat;
  float *OutFloatFromInt, *OutFloatFromDouble;
  gpuErrCheck(gpuMallocManaged(&OutDoubleFromFloat, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&OutIntFromDouble, sizeof(int)));
  gpuErrCheck(gpuMallocManaged(&OutFloatFromInt, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&OutDoubleFromInt, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&OutIntFromFloat, sizeof(int)));
  gpuErrCheck(gpuMallocManaged(&OutFloatFromDouble, sizeof(float)));

  gpuErrCheck(KernelHandle.launch(
      {1, 1, 1}, {1, 1, 1}, 0, nullptr, OutDoubleFromFloat, OutIntFromDouble,
      OutFloatFromInt, OutDoubleFromInt, OutIntFromFloat, OutFloatFromDouble));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "DoubleFromFloat = " << *OutDoubleFromFloat << "\n";
  std::cout << "IntFromDouble = " << *OutIntFromDouble << "\n";
  std::cout << "FloatFromInt = " << *OutFloatFromInt << "\n";
  std::cout << "DoubleFromInt = " << *OutDoubleFromInt << "\n";
  std::cout << "IntFromFloat = " << *OutIntFromFloat << "\n";
  std::cout << "FloatFromDouble = " << *OutFloatFromDouble << "\n";

  gpuErrCheck(gpuFree(OutDoubleFromFloat));
  gpuErrCheck(gpuFree(OutIntFromDouble));
  gpuErrCheck(gpuFree(OutFloatFromInt));
  gpuErrCheck(gpuFree(OutDoubleFromInt));
  gpuErrCheck(gpuFree(OutIntFromFloat));
  gpuErrCheck(gpuFree(OutFloatFromDouble));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: DoubleFromFloat = 2.5
// CHECK-NEXT: IntFromDouble = 3
// CHECK-NEXT: FloatFromInt = -7
// CHECK-NEXT: DoubleFromInt = 3
// CHECK-NEXT: IntFromFloat = -7
// CHECK-NEXT: FloatFromDouble = 2.5
// CHECK-FIRST: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
