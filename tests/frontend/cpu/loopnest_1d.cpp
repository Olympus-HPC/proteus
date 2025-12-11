// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_1d | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_1d | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto get1DLoopNestFunction(int N, int TileSize) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *, double *)>("loopnest_1d");

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
        .tile(TileSize)
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

static auto get1DSimpleLoopNestFunction(int N) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *, double *)>("loopnest_1d_simple");

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

    // Test non-tiled version.
    F.forLoop(I, Zero, UB, IncOne, [&]() { A[I] = B[I] * 3.0; }).emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int N = 8;
  constexpr int TileSize = 4;

  // Test 1D tiled loop.
  auto [JitMod1, F1] = get1DLoopNestFunction(N, TileSize);
  JitMod1->compile();

  double *A1 = new double[N];
  double *B1 = new double[N];

  for (int I = 0; I < N; I++) {
    B1[I] = I + 1;
    A1[I] = 0.0;
  }

  F1(A1, B1);

  std::cout << "1D Tiled Results:\n";
  for (int I = 0; I < N; I++) {
    std::cout << "A1[" << I << "] = " << A1[I] << "\n";
  }

  // Test 1D simple (non-tiled) loop.
  auto [JitMod2, F2] = get1DSimpleLoopNestFunction(N);
  JitMod2->compile();

  double *A2 = new double[N];
  double *B2 = new double[N];

  for (int I = 0; I < N; I++) {
    B2[I] = I + 1;
    A2[I] = 0.0;
  }

  F2(A2, B2);

  std::cout << "1D Simple Results:\n";
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
// CHECK: 1D Tiled Results:
// CHECK-NEXT: A1[0] = 3
// CHECK-NEXT: A1[1] = 6
// CHECK-NEXT: A1[2] = 9
// CHECK-NEXT: A1[3] = 12
// CHECK-NEXT: A1[4] = 15
// CHECK-NEXT: A1[5] = 18
// CHECK-NEXT: A1[6] = 21
// CHECK-NEXT: A1[7] = 24
// CHECK: 1D Simple Results:
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
