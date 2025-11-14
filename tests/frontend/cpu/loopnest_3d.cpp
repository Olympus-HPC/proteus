// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_3d | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_3d | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto get3DLoopNestFunction(int DI, int DJ, int DK, int TileI, int TileJ,
                                  int TileK) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *, double *)>("loopnest_3d");

  auto I = F.declVar<int>("i");
  auto J = F.declVar<int>("j");
  auto K = F.declVar<int>("k");
  auto IncOne = F.declVar<int>("inc");
  auto UBI = F.declVar<int>("ubi");
  auto UBJ = F.declVar<int>("ubj");
  auto UBK = F.declVar<int>("ubk");

  auto &A = F.getArg<0>();
  auto &B = F.getArg<1>();

  F.beginFunction();
  {
    I = 0;
    J = 0;
    K = 0;
    UBI = DI;
    UBJ = DJ;
    UBK = DK;
    IncOne = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;
    auto RowBias = F.defVar<int>(0, "row_bias");

    F.buildLoopNest(
         F.forLoop<int>({I, Zero, UBI, IncOne}).tile(TileI),
         F.forLoop<int>({J, Zero, UBJ, IncOne}, [&]() { RowBias = J; })
             .tile(TileJ),
         F.forLoop<int>({K, Zero, UBK, IncOne},
                        [&]() {
                          auto Idx = I * DJ * DK + J * DK + K;
                          A[Idx] = B[Idx] + I + J + K + RowBias;
                        })
             .tile(TileK))
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

static auto get3DUniformTileFunction(int DI, int DJ, int DK, int TileSize) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F =
      JitMod->addFunction<void(double *, double *)>("loopnest_3d_uniform");

  auto I = F.declVar<int>("i");
  auto J = F.declVar<int>("j");
  auto K = F.declVar<int>("k");
  auto IncOne = F.declVar<int>("inc");
  auto UBI = F.declVar<int>("ubi");
  auto UBJ = F.declVar<int>("ubj");
  auto UBK = F.declVar<int>("ubk");

  auto &A = F.getArg<0>();
  auto &B = F.getArg<1>();

  F.beginFunction();
  {
    I = 0;
    J = 0;
    K = 0;
    UBI = DI;
    UBJ = DJ;
    UBK = DK;
    IncOne = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;
    auto RowBias = F.defVar<int>(0, "row_bias");

    F.buildLoopNest(
         F.forLoop<int>({I, Zero, UBI, IncOne}).tile(TileSize),
         F.forLoop<int>({J, Zero, UBJ, IncOne}, [&]() { RowBias = J; })
             .tile(TileSize),
         F.forLoop<int>({K, Zero, UBK, IncOne},
                        [&]() {
                          auto Idx = I * DJ * DK + J * DK + K;
                          A[Idx] = B[Idx] + I + J + K + RowBias;
                        })
             .tile(TileSize))
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int DI = 4, DJ = 2, DK = 2;
  constexpr int TileI = 2, TileJ = 1, TileK = 2;
  constexpr int UniformTile = 2;
  constexpr int SIZE = DI * DJ * DK;

  // Test 3D variadic tiling.
  auto [JitMod1, F1] = get3DLoopNestFunction(DI, DJ, DK, TileI, TileJ, TileK);
  JitMod1->compile();

  double *A1 = new double[SIZE];
  double *B1 = new double[SIZE];

  for (int I = 0; I < SIZE; I++) {
    B1[I] = I;
    A1[I] = 0.0;
  }

  F1(A1, B1);

  std::cout << "3D Variadic Tiling Results:\n";
  for (int I = 0; I < SIZE; I++) {
    std::cout << "A1[" << I << "] = " << A1[I] << "\n";
  }

  // Test 3D uniform tiling.
  auto [JitMod2, F2] = get3DUniformTileFunction(DI, DJ, DK, UniformTile);
  JitMod2->compile();

  double *A2 = new double[SIZE];
  double *B2 = new double[SIZE];

  for (int I = 0; I < SIZE; I++) {
    B2[I] = I;
    A2[I] = 0.0;
  }

  F2(A2, B2);

  std::cout << "3D Uniform Tiling Results:\n";
  for (int I = 0; I < SIZE; I++) {
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
// CHECK: 3D Variadic Tiling Results:
// CHECK-NEXT: A1[0] = 0
// CHECK-NEXT: A1[1] = 2
// CHECK-NEXT: A1[2] = 4
// CHECK-NEXT: A1[3] = 6
// CHECK-NEXT: A1[4] = 5
// CHECK-NEXT: A1[5] = 7
// CHECK-NEXT: A1[6] = 9
// CHECK-NEXT: A1[7] = 11
// CHECK-NEXT: A1[8] = 10
// CHECK-NEXT: A1[9] = 12
// CHECK-NEXT: A1[10] = 14
// CHECK-NEXT: A1[11] = 16
// CHECK-NEXT: A1[12] = 15
// CHECK-NEXT: A1[13] = 17
// CHECK-NEXT: A1[14] = 19
// CHECK-NEXT: A1[15] = 21
// CHECK: 3D Uniform Tiling Results:
// CHECK-NEXT: A2[0] = 0
// CHECK-NEXT: A2[1] = 2
// CHECK-NEXT: A2[2] = 4
// CHECK-NEXT: A2[3] = 6
// CHECK-NEXT: A2[4] = 5
// CHECK-NEXT: A2[5] = 7
// CHECK-NEXT: A2[6] = 9
// CHECK-NEXT: A2[7] = 11
// CHECK-NEXT: A2[8] = 10
// CHECK-NEXT: A2[9] = 12
// CHECK-NEXT: A2[10] = 14
// CHECK-NEXT: A2[11] = 16
// CHECK-NEXT: A2[12] = 15
// CHECK-NEXT: A2[13] = 17
// CHECK-NEXT: A2[14] = 19
// CHECK-NEXT: A2[15] = 21
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 2 accesses 2
