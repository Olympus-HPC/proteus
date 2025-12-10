// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_unroll | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_unroll | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

// Test unroll() without count.
static auto getUnrollEnableFunction(int N) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *, double *)>("unroll_enable");

  auto I = F.declVar<int>("i");
  auto IncOne = F.declVar<int>("inc");
  auto UB = F.declVar<int>("ub");

  auto &A = F.getArg<0>();
  auto &B = F.getArg<1>();

  F.beginFunction();
  {
    I = 0;
    UB = N;
    IncOne = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;

    F.forLoop(I, Zero, UB, IncOne, [&]() { A[I] = B[I] * 2.0; })
        .unroll()
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

// Test unroll(count) with specific count.
static auto getUnrollCountFunction(int N) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *, double *)>("unroll_count");

  auto I = F.declVar<int>("i");
  auto IncOne = F.declVar<int>("inc");
  auto UB = F.declVar<int>("ub");

  auto &A = F.getArg<0>();
  auto &B = F.getArg<1>();

  F.beginFunction();
  {
    I = 0;
    UB = N;
    IncOne = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;

    F.forLoop(I, Zero, UB, IncOne, [&]() { A[I] = B[I] * 3.0; })
        .unroll(4)
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int N = 8;

  auto [JitMod1, F1] = getUnrollEnableFunction(N);
  std::cout << "=== Unroll Enable IR ===\n" << std::flush;
  JitMod1->print();
  JitMod1->compile();

  double *A1 = new double[N];
  double *B1 = new double[N];

  for (int I = 0; I < N; I++) {
    B1[I] = I + 1;
    A1[I] = 0.0;
  }

  F1(A1, B1);

  std::cout << "Unroll Enable Results:\n";
  for (int I = 0; I < N; I++) {
    std::cout << "A1[" << I << "] = " << A1[I] << "\n";
  }

  auto [JitMod2, F2] = getUnrollCountFunction(N);
  std::cout << "=== Unroll Count IR ===\n" << std::flush;
  JitMod2->print();
  JitMod2->compile();

  double *A2 = new double[N];
  double *B2 = new double[N];

  for (int I = 0; I < N; I++) {
    B2[I] = I + 1;
    A2[I] = 0.0;
  }

  F2(A2, B2);

  std::cout << "Unroll Count Results:\n";
  for (int I = 0; I < N; I++) {
    std::cout << "A2[" << I << "] = " << A2[I] << "\n";
  }

  delete[] A1;
  delete[] B1;
  delete[] A2;
  delete[] B2;

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: === Unroll Enable IR ===
// CHECK: br {{.*}}!llvm.loop [[LOOP1:![0-9]+]]
// CHECK: [[LOOP1]] = distinct !{[[LOOP1]], [[UNROLL_ENABLE1:![0-9]+]]}
// CHECK: [[UNROLL_ENABLE1]] = !{!"llvm.loop.unroll.enable"}
// CHECK: Unroll Enable Results:
// CHECK-NEXT: A1[0] = 2
// CHECK-NEXT: A1[1] = 4
// CHECK-NEXT: A1[2] = 6
// CHECK-NEXT: A1[3] = 8
// CHECK-NEXT: A1[4] = 10
// CHECK-NEXT: A1[5] = 12
// CHECK-NEXT: A1[6] = 14
// CHECK-NEXT: A1[7] = 16
// CHECK: === Unroll Count IR ===
// CHECK: br {{.*}}!llvm.loop [[LOOP2:![0-9]+]]
// CHECK: [[LOOP2]] = distinct !{[[LOOP2]], [[UNROLL_ENABLE2:![0-9]+]], [[UNROLL_COUNT2:![0-9]+]]}
// CHECK: [[UNROLL_ENABLE2]] = !{!"llvm.loop.unroll.enable"}
// CHECK: [[UNROLL_COUNT2]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: Unroll Count Results:
// CHECK-NEXT: A2[0] = 3
// CHECK-NEXT: A2[1] = 6
// CHECK-NEXT: A2[2] = 9
// CHECK-NEXT: A2[3] = 12
// CHECK-NEXT: A2[4] = 15
// CHECK-NEXT: A2[5] = 18
// CHECK-NEXT: A2[6] = 21
// CHECK-NEXT: A2[7] = 24
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 2 accesses 2
