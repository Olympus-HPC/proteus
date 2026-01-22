// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/vars.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/vars.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <iostream>

#include <proteus/JitFrontend.h>
#include <proteus/JitInterface.h>

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

  // Test declVars with anonymous variables.
  {
    auto J = proteus::JitModule(TARGET);
    auto KernelHandle = J.addKernel<void(double *)>("declVarsAnonymous");
    auto &F = KernelHandle.F;

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.declVars<int, double, float>();
      I = 10;
      D = 20.5;
      Fl = 30.25f;
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    J.compile();

    double *Result;
    gpuErrCheck(gpuMallocManaged(&Result, sizeof(double) * 3));
    Result[0] = 0.0;
    Result[1] = 0.0;
    Result[2] = 0.0;

    gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Result));
    gpuErrCheck(gpuDeviceSynchronize());

    std::cout << "declVarsAnon[0] = " << Result[0] << "\n";
    std::cout << "declVarsAnon[1] = " << Result[1] << "\n";
    std::cout << "declVarsAnon[2] = " << Result[2] << "\n";

    gpuErrCheck(gpuFree(Result));
  }

  // Test declVars with named variables.
  {
    auto J = proteus::JitModule(TARGET);
    auto KernelHandle = J.addKernel<void(double *)>("declVarsNamed");
    auto &F = KernelHandle.F;

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.declVars<int, double, float>("I", "D", "F");
      I = 100;
      D = 200.5;
      Fl = 300.25f;
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double *Result;
    gpuErrCheck(gpuMallocManaged(&Result, sizeof(double) * 3));
    Result[0] = 0.0;
    Result[1] = 0.0;
    Result[2] = 0.0;

    gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Result));
    gpuErrCheck(gpuDeviceSynchronize());

    std::cout << "declVarsNamed[0] = " << Result[0] << "\n";
    std::cout << "declVarsNamed[1] = " << Result[1] << "\n";
    std::cout << "declVarsNamed[2] = " << Result[2] << "\n";

    gpuErrCheck(gpuFree(Result));
  }

  // Test defVars with raw values.
  {
    auto J = proteus::JitModule(TARGET);
    auto KernelHandle = J.addKernel<void(double *)>("defVarsRaw");
    auto &F = KernelHandle.F;

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.defVars(10, 20.5, 30.25f);
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    J.compile();

    double *Result;
    gpuErrCheck(gpuMallocManaged(&Result, sizeof(double) * 3));
    Result[0] = 0.0;
    Result[1] = 0.0;
    Result[2] = 0.0;

    gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Result));
    gpuErrCheck(gpuDeviceSynchronize());

    std::cout << "defVarsRaw[0] = " << Result[0] << "\n";
    std::cout << "defVarsRaw[1] = " << Result[1] << "\n";
    std::cout << "defVarsRaw[2] = " << Result[2] << "\n";

    gpuErrCheck(gpuFree(Result));
  }

  // Test defVars with named pairs.
  {
    auto J = proteus::JitModule(TARGET);
    auto KernelHandle = J.addKernel<void(double *)>("defVarsNamed");
    auto &F = KernelHandle.F;

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.defVars(std::pair{50, "I"}, std::pair{60.5, "D"},
                                  std::pair{70.25f, "F"});
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double *Result;
    gpuErrCheck(gpuMallocManaged(&Result, sizeof(double) * 3));
    Result[0] = 0.0;
    Result[1] = 0.0;
    Result[2] = 0.0;

    gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Result));
    gpuErrCheck(gpuDeviceSynchronize());

    std::cout << "defVarsNamed[0] = " << Result[0] << "\n";
    std::cout << "defVarsNamed[1] = " << Result[1] << "\n";
    std::cout << "defVarsNamed[2] = " << Result[2] << "\n";

    gpuErrCheck(gpuFree(Result));
  }

  // Test defVars with mixed raw and named.
  {
    auto J = proteus::JitModule(TARGET);
    auto KernelHandle = J.addKernel<void(double *)>("defVarsMixed");
    auto &F = KernelHandle.F;

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D] = F.defVars(123, std::pair{456.789, "D"});
      Arg[0] = I;
      Arg[1] = D;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double *Result;
    gpuErrCheck(gpuMallocManaged(&Result, sizeof(double) * 2));
    Result[0] = 0.0;
    Result[1] = 0.0;

    gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Result));
    gpuErrCheck(gpuDeviceSynchronize());

    std::cout << "defVarsMixed[0] = " << Result[0] << "\n";
    std::cout << "defVarsMixed[1] = " << Result[1] << "\n";

    gpuErrCheck(gpuFree(Result));
  }

  proteus::finalize();
  return 0;
}

// clang-format off

// CHECK: declVarsAnon[0] = 10
// CHECK-NEXT: declVarsAnon[1] = 20.5
// CHECK-NEXT: declVarsAnon[2] = 30.25

// CHECK: define {{.*}} @declVarsNamed
// CHECK-DAG: %I = alloca i32
// CHECK-DAG: %D = alloca double
// CHECK-DAG: %F = alloca float
// CHECK: declVarsNamed[0] = 100
// CHECK-NEXT: declVarsNamed[1] = 200.5
// CHECK-NEXT: declVarsNamed[2] = 300.25

// CHECK: defVarsRaw[0] = 10
// CHECK-NEXT: defVarsRaw[1] = 20.5
// CHECK-NEXT: defVarsRaw[2] = 30.25

// CHECK: define {{.*}} @defVarsNamed
// CHECK-DAG: %I = alloca i32
// CHECK-DAG: %D = alloca double
// CHECK-DAG: %F = alloca float
// CHECK: defVarsNamed[0] = 50
// CHECK-NEXT: defVarsNamed[1] = 60.5
// CHECK-NEXT: defVarsNamed[2] = 70.25

// CHECK: define {{.*}} @defVarsMixed
// CHECK-DAG: %D = alloca double
// CHECK: defVarsMixed[0] = 123
// CHECK-NEXT: defVarsMixed[1] = 456.789

// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 5
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 5 accesses 5
