// RUN: rm -rf .proteus
// RUN: ./assign | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule("cuda");
  auto &F = J.addKernel<double *>("Assign");
  auto &A = F.declVar<double>("a");
  auto &Tid = F.declVar<int>("tid");
  auto &Arg0 = F.getArg(0);
  F.beginFunction();
  {
    Tid = F.callBuiltin(proteus::builtins::cuda::getThreadIdX);
    Arg0 = 42;
    F.ret();
  }
  F.endFunction();

  J.print();
  J.compileForDevice();

  double *KernelArg0;
  cudaMallocManaged(&KernelArg0, sizeof(double));
  proteusCudaErrCheck(J.launch(F, {1}, {1}, {&KernelArg0}, 0, nullptr));

  cudaDeviceSynchronize();

  std::cout << "KernelArg0 " << (*KernelArg0) << "\n";

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
