// RUN: rm -rf .proteus
// RUN: ./loopnest_1d | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

auto get1DLoopNestFunction(int N, int TILE_SIZE) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void, double *, double *>("loopnest_1d");

  auto &I = F.declVar<int>("i");
  auto &IncOne = F.declVar<int>("inc");
  auto &UB = F.declVar<int>("ub");

  auto args = F.getArgs();
  auto &A = std::get<0>(args);
  auto &B = std::get<1>(args);

  F.beginFunction();
  {
    I = 0;
    UB = N;
    IncOne = 1;
    auto &Zero = F.declVar<int>("zero");
    Zero = 0;

    F.LoopNest({F.ForLoop({I, Zero, UB, IncOne}, [&]() { A[I] = B[I] * 3.0; })
                    .tile(TILE_SIZE)})
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

auto get1DSimpleLoopNestFunction(int N) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void, double *, double *>("loopnest_1d_simple");

  auto &I = F.declVar<int>("i");
  auto &IncOne = F.declVar<int>("inc");
  auto &UB = F.declVar<int>("ub");

  auto args = F.getArgs();
  auto &A = std::get<0>(args);
  auto &B = std::get<1>(args);

  F.beginFunction();
  {
    I = 0;
    UB = N;
    IncOne = 1;
    auto &Zero = F.declVar<int>("zero");
    Zero = 0;

    // Test non-tiled version
    F.LoopNest({F.ForLoop({I, Zero, UB, IncOne}, [&]() { A[I] = B[I] * 3.0; })})
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int N = 8;
  constexpr int TILE_SIZE = 4;

  // Test 1D tiled loop
  auto [JitMod1, F1] = get1DLoopNestFunction(N, TILE_SIZE);
  JitMod1->compile();

  double *A1 = new double[N];
  double *B1 = new double[N];

  for (int i = 0; i < N; i++) {
    B1[i] = i + 1;
    A1[i] = 0.0;
  }

  F1(A1, B1);

  std::cout << "1D Tiled Results:\n";
  for (int i = 0; i < N; i++) {
    std::cout << "A1[" << i << "] = " << A1[i] << "\n";
  }

  // Test 1D simple (non-tiled) loop
  auto [JitMod2, F2] = get1DSimpleLoopNestFunction(N);
  JitMod2->compile();

  double *A2 = new double[N];
  double *B2 = new double[N];

  for (int i = 0; i < N; i++) {
    B2[i] = i + 1;
    A2[i] = 0.0;
  }

  F2(A2, B2);

  std::cout << "1D Simple Results:\n";
  for (int i = 0; i < N; i++) {
    std::cout << "A2[" << i << "] = " << A2[i] << "\n";
  }

  delete[] A1;
  delete[] B1;
  delete[] A2;
  delete[] B2;

  proteus::finalize();
  return 0;
}

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