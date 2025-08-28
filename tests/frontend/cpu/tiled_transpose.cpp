// RUN: rm -rf .proteus
// RUN: ./tiled_transpose | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto getTiled2DTransposeFunction(int ROWS, int COLS, int TileSize) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  // TODO: Should JitModule handle de-duplicate function names?
  static int Counter = 0;
  auto &F = JitMod->addFunction<void, double *, double *>(
      "tiled_transpose_" + std::to_string(Counter++));

  auto &I = F.declVar<int>("i");
  auto &J = F.declVar<int>("j");
  auto &IncOne = F.declVar<int>("inc");
  auto &UBRows = F.declVar<int>("ub_rows");
  auto &UBCols = F.declVar<int>("ub_cols");

  auto Args = F.getArgs();
  auto &A = std::get<0>(Args);
  auto &B = std::get<1>(Args);

  F.beginFunction();
  {
    I = 0;
    J = 0;
    UBRows = ROWS;
    UBCols = COLS;
    IncOne = 1;
    auto &Zero = F.declVar<int>("zero");
    Zero = 0;

    F.buildLoopNest(
         {F.forLoop({I, Zero, UBRows, IncOne}),
          F.forLoop({J, Zero, UBCols, IncOne},
                                 [&]() {
                                   auto AIdx = J * ROWS + I;
                                   auto BIdx = I * COLS + J;
                                   A[AIdx] = B[BIdx];
                                 })
              }).tile(TileSize).emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  // tile size divides both ROWS and COLS
  {
    constexpr int ROWS = 4;
    constexpr int COLS = 4;
    constexpr int TileSize = 2;
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
  }

  // tile size does not divide ROWS or COLS
  {
    constexpr int ROWS = 5;
    constexpr int COLS = 4;
    constexpr int TileSize = 3;
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
  }

  proteus::finalize();
  return 0;
}

// CHECK: Input B:
// CHECK-NEXT: 0 1 2 3
// CHECK-NEXT: 4 5 6 7
// CHECK-NEXT: 8 9 10 11
// CHECK-NEXT: 12 13 14 15
// CHECK: Transposed A:
// CHECK-NEXT: 0 4 8 12
// CHECK-NEXT: 1 5 9 13
// CHECK-NEXT: 2 6 10 14
// CHECK-NEXT: 3 7 11 15
// CHECK: Input B:
// CHECK-NEXT: 0 1 2 3
// CHECK-NEXT: 4 5 6 7
// CHECK-NEXT: 8 9 10 11
// CHECK-NEXT: 12 13 14 15
// CHECK-NEXT: 16 17 18 19
// CHECK: Transposed A:
// CHECK-NEXT: 0 4 8 12 16
// CHECK-NEXT: 1 5 9 13 17
// CHECK-NEXT: 2 6 10 14 18
// CHECK-NEXT: 3 7 11 15 19