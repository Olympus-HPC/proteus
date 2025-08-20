// RUN: rm -rf .proteus
// RUN: ./loopnest_3d | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

auto get3DLoopNestFunction(int DI, int DJ, int DK, int TILE_I, int TILE_J,
                           int TILE_K) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void, double *, double *>("loopnest_3d");

  auto &I = F.declVar<int>("i");
  auto &J = F.declVar<int>("j");
  auto &K = F.declVar<int>("k");
  auto &IncOne = F.declVar<int>("inc");
  auto &UBI = F.declVar<int>("ubi");
  auto &UBJ = F.declVar<int>("ubj");
  auto &UBK = F.declVar<int>("ubk");

  auto args = F.getArgs();
  auto &A = std::get<0>(args);
  auto &B = std::get<1>(args);

  F.beginFunction();
  {
    I = 0;
    J = 0;
    K = 0;
    UBI = DI;
    UBJ = DJ;
    UBK = DK;
    IncOne = 1;
    auto &Zero = F.declVar<int>("zero");
    Zero = 0;

    F.LoopNest({F.ForLoop({I, Zero, UBI, IncOne}).tile(TILE_I),
                F.ForLoop({J, Zero, UBJ, IncOne}).tile(TILE_J),
                F.ForLoop({K, Zero, UBK, IncOne},
                          [&]() {
                            auto idx = I * DJ * DK + J * DK + K;
                            A[idx] = B[idx] + I + J + K;
                          })
                    .tile(TILE_K)})
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

auto get3DUniformTileFunction(int DI, int DJ, int DK, int TILE_SIZE) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F =
      JitMod->addFunction<void, double *, double *>("loopnest_3d_uniform");

  auto &I = F.declVar<int>("i");
  auto &J = F.declVar<int>("j");
  auto &K = F.declVar<int>("k");
  auto &IncOne = F.declVar<int>("inc");
  auto &UBI = F.declVar<int>("ubi");
  auto &UBJ = F.declVar<int>("ubj");
  auto &UBK = F.declVar<int>("ubk");

  auto args = F.getArgs();
  auto &A = std::get<0>(args);
  auto &B = std::get<1>(args);

  F.beginFunction();
  {
    I = 0;
    J = 0;
    K = 0;
    UBI = DI;
    UBJ = DJ;
    UBK = DK;
    IncOne = 1;
    auto &Zero = F.declVar<int>("zero");
    Zero = 0;

    F.LoopNest({F.ForLoop({I, Zero, UBI, IncOne}).tile(TILE_SIZE),
                F.ForLoop({J, Zero, UBJ, IncOne}).tile(TILE_SIZE),
                F.ForLoop({K, Zero, UBK, IncOne},
                          [&]() {
                            auto idx = I * DJ * DK + J * DK + K;
                            A[idx] = B[idx] * 2;
                          })
                    .tile(TILE_SIZE)})
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int DI = 4, DJ = 2, DK = 2;
  constexpr int TILE_I = 2, TILE_J = 1, TILE_K = 2;
  constexpr int UNIFORM_TILE = 2;
  constexpr int SIZE = DI * DJ * DK;

  // Test 3D variadic tiling
  auto [JitMod1, F1] =
      get3DLoopNestFunction(DI, DJ, DK, TILE_I, TILE_J, TILE_K);
  JitMod1->compile();

  double *A1 = new double[SIZE];
  double *B1 = new double[SIZE];

  for (int i = 0; i < SIZE; i++) {
    B1[i] = i;
    A1[i] = 0.0;
  }

  F1(A1, B1);

  std::cout << "3D Variadic Tiling Results:\n";
  for (int i = 0; i < SIZE; i++) {
    std::cout << "A1[" << i << "] = " << A1[i] << "\n";
  }

  // Test 3D uniform tiling
  auto [JitMod2, F2] = get3DUniformTileFunction(DI, DJ, DK, UNIFORM_TILE);
  JitMod2->compile();

  double *A2 = new double[SIZE];
  double *B2 = new double[SIZE];

  for (int i = 0; i < SIZE; i++) {
    B2[i] = i;
    A2[i] = 0.0;
  }

  F2(A2, B2);

  std::cout << "3D Uniform Tiling Results:\n";
  for (int i = 0; i < SIZE; i++) {
    std::cout << "A2[" << i << "] = " << A2[i] << "\n";
  }

  delete[] A1;
  delete[] B1;
  delete[] A2;
  delete[] B2;

  proteus::finalize();
  return 0;
}

// CHECK: 3D Variadic Tiling Results:
// CHECK-NEXT: A1[0] = 0
// CHECK-NEXT: A1[1] = 2
// CHECK-NEXT: A1[2] = 3
// CHECK-NEXT: A1[3] = 5
// CHECK-NEXT: A1[4] = 5
// CHECK-NEXT: A1[5] = 7
// CHECK-NEXT: A1[6] = 8
// CHECK-NEXT: A1[7] = 10
// CHECK-NEXT: A1[8] = 10
// CHECK-NEXT: A1[9] = 12
// CHECK-NEXT: A1[10] = 13
// CHECK-NEXT: A1[11] = 15
// CHECK-NEXT: A1[12] = 15
// CHECK-NEXT: A1[13] = 17
// CHECK-NEXT: A1[14] = 18
// CHECK-NEXT: A1[15] = 20
// CHECK: 3D Uniform Tiling Results:
// CHECK-NEXT: A2[0] = 0
// CHECK-NEXT: A2[1] = 2
// CHECK-NEXT: A2[2] = 4
// CHECK-NEXT: A2[3] = 6
// CHECK-NEXT: A2[4] = 8
// CHECK-NEXT: A2[5] = 10
// CHECK-NEXT: A2[6] = 12
// CHECK-NEXT: A2[7] = 14
// CHECK-NEXT: A2[8] = 16
// CHECK-NEXT: A2[9] = 18
// CHECK-NEXT: A2[10] = 20
// CHECK-NEXT: A2[11] = 22
// CHECK-NEXT: A2[12] = 24
// CHECK-NEXT: A2[13] = 26
// CHECK-NEXT: A2[14] = 28
// CHECK-NEXT: A2[15] = 30