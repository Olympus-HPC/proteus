// RUN: rm -rf .proteus
// RUN: ./assign | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>
#include <proteus/Utils.h>

#include <hip/hip_runtime.h>

using namespace proteus;

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule("hip");
  auto KernelHandle = J.addKernel<float *, size_t>("assign");
  auto &F = KernelHandle.F;
  auto &i = F.declVar<size_t>("i");
  auto &j = F.declVar<size_t>("j");
  auto &totThreads = F.declVar<size_t>("totThreads");
  auto &p = F.getArg(0);
  auto &vector_size = F.getArg(1);
  auto &V1 = F.declVar<float>("v1");
  auto &V2 = F.declVar<size_t>("v2");
  F.beginFunction();
  {
    // i = F.callBuiltin(builtins::hip::getBlockIdX) *
    //         F.callBuiltin(builtins::hip::getBlockDimX) +
    //     F.callBuiltin(builtins::hip::getThreadIdX);
    V2 = 2;
    V1 = 1 + 2.0f * V2;
    F.beginIf(V1 >= 2.0);
    { p[0] = 42; }
    F.endIf();
    p[1] = V1;

    // totThreads = F.callBuiltin(builtins::hip::getGridDimX) *
    //              F.callBuiltin(builtins::hip::getBlockDimX);
    // F.beginLoop(j, i, vector_size, totThreads);
    //{ p[j] = vector_size; }
    // F.endLoop();
    F.ret();
  }
  F.endFunction();

  J.print();
  J.compile();

  constexpr int Size = 10000;
  float *KernelArg0;
  proteusHipErrCheck(hipMallocManaged(&KernelArg0, sizeof(float) * Size));
  for (int i = 0; i < 2; ++i) {
    KernelArg0[i] = 0.0;
    std::cout << "X[" << i << "] = " << KernelArg0[i] << "\n";
  }
  std::cout << "===\n";
  const int threadsPerBlock = 256;
  const dim3 grids((Size + threadsPerBlock - 1) / threadsPerBlock);
  const dim3 blocks(threadsPerBlock);
  std::cout << "grids " << grids.x << " blocks " << blocks.x << "\n";

  auto KernelHandle2 = J.getKernelHandle<float *, size_t>("assign");
  proteusHipErrCheck(KernelHandle2.launch({grids.x, 1, 1}, {blocks.x, 1, 1}, 0,
                                          0, KernelArg0, Size));
  proteusHipErrCheck(hipDeviceSynchronize());

  for (int i = 0; i < 2; ++i) {
    std::cout << "X'[" << i << "] = " << KernelArg0[i] << "\n";
  }

  proteus::finalize();
  return 0;
}

// clang-format off

// CHECK: ; ModuleID = 'JitModule'
// CHECK: source_filename = "JitModule"
// CHECK: target triple = "arm64-apple-darwin22.6.0"
// CHECK-EMPTY: 
// CHECK-LABEL: define double @Assignments(ptr %0) {
// CHECK: entry:
// CHECK:   %res. = alloca ptr, align 8
// CHECK:   %a = alloca double, align 8
// CHECK:   %arg.0 = alloca ptr, align 8
// CHECK:   store ptr %0, ptr %arg.0, align 8
// CHECK:   br label %body
// CHECK-EMPTY: 
// CHECK: body:                                             ; preds = %entry
// CHECK:   store double 1.000000e+00, ptr %a, align 8
// CHECK:   %1 = load ptr, ptr %arg.0, align 8
// CHECK:   %2 = getelementptr inbounds double, ptr %1, i64 0
// CHECK:   store ptr %2, ptr %res., align 8
// CHECK:   %3 = load ptr, ptr %res., align 8
// CHECK:   store double 4.200000e+01, ptr %3, align 8
// CHECK:   %4 = load double, ptr %a, align 8
// CHECK:   ret double %4
// CHECK: }

// CHECK: Ret 1
// CHECK: X[0] = 42
