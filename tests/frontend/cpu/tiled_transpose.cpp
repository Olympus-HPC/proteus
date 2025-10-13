// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/tiled_transpose 4 4 2 | %FILECHECK %s --check-prefixes=FIRST,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/tiled_transpose 4 4 2 | %FILECHECK %s --check-prefixes=FIRST,CHECK-SECOND
// Third run does not use the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/tiled_transpose 5 4 3 | %FILECHECK %s --check-prefixes=THIRD,CHECK-THIRD
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdlib>
#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto getTiled2DTransposeFunction(int ROWS, int COLS, int TileSize) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  static int Counter = 0;
  auto &F = JitMod->addFunction<void(double *, double *)>(
      "tiled_transpose_" + std::to_string(Counter++));

  auto I = F.declVarTT<int>("i");
  auto J = F.declVarTT<int>("j");
  auto IncOne = F.declVarTT<int>("inc");
  auto UBRows = F.declVarTT<int>("ub_rows");
  auto UBCols = F.declVarTT<int>("ub_cols");

  auto Args = F.getArgsTT();
  auto &A = std::get<0>(Args);
  auto &B = std::get<1>(Args);

  F.beginFunction();
  {
    I = 0;
    J = 0;
    UBRows = ROWS;
    UBCols = COLS;
    IncOne = 1;
    auto Zero = F.declVarTT<int>("zero");
    Zero = 0;

    F.buildLoopNest(F.forLoop<int>({I, Zero, UBRows, IncOne}),
                    F.forLoop<int>({J, Zero, UBCols, IncOne},
                              [&]() {
                                auto AIdx = J * ROWS + I;
                                auto BIdx = I * COLS + J;
                                A[AIdx] = B[BIdx];
                              }))
        .tile(TileSize)
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main(int argc, char **argv) {
  proteus::init();

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <ROWS> <COLS> <TileSize>\n";
    proteus::finalize();
    return 1;
  }

  int ROWS = std::atoi(argv[1]);
  int COLS = std::atoi(argv[2]);
  int TileSize = std::atoi(argv[3]);

  auto [JitMod, F] = getTiled2DTransposeFunction(ROWS, COLS, TileSize);

  JitMod->compile();

  double *A = (double *)new double[ROWS * COLS];
  double *B = (double *)new double[ROWS * COLS];

  for (int I = 0; I < ROWS; I++) {
    for (int J = 0; J < COLS; J++) {
      B[I * COLS + J] = I * COLS + J;
      A[I * COLS + J] = 0.0;
    }
  }

  std::cout << "Input B:\n";
  for (int I = 0; I < ROWS; I++) {
    for (int J = 0; J < COLS; J++) {
      std::cout << B[I * COLS + J] << " ";
    }
    std::cout << "\n";
  }

  F(A, B);

  std::cout << "Transposed A:\n";
  for (int I = 0; I < COLS; I++) {
    for (int J = 0; J < ROWS; J++) {
      std::cout << A[I * ROWS + J] << " ";
    }
    std::cout << "\n";
  }

  delete[] A;
  delete[] B;

  proteus::finalize();
  return 0;
}

// FIRST: Input B:
// FIRST-NEXT: 0 1 2 3
// FIRST-NEXT: 4 5 6 7
// FIRST-NEXT: 8 9 10 11
// FIRST-NEXT: 12 13 14 15
// FIRST: Transposed A:
// FIRST-NEXT: 0 4 8 12
// FIRST-NEXT: 1 5 9 13
// FIRST-NEXT: 2 6 10 14
// FIRST-NEXT: 3 7 11 15
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// THIRD: Input B:
// THIRD-NEXT: 0 1 2 3
// THIRD-NEXT: 4 5 6 7
// THIRD-NEXT: 8 9 10 11
// THIRD-NEXT: 12 13 14 15
// THIRD-NEXT: 16 17 18 19
// THIRD: Transposed A:
// THIRD-NEXT: 0 4 8 12 16
// THIRD-NEXT: 1 5 9 13 17
// THIRD-NEXT: 2 6 10 14 18
// THIRD-NEXT: 3 7 11 15 19
// CHECK-THIRD: JitStorageCache hits 0 total 1
