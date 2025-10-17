// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/tiled_matmul 16 3 4 5 | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/tiled_matmul 16 3 4 5 | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdlib>
#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto getTiledMatmulFunction(int N, int TileI, int TileJ, int TileK) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F =
      JitMod->addFunction<void(double *, double *, double *)>("tiled_matmul");
  {

    auto &C = F.getArg<0>();
    auto &A = F.getArg<1>();
    auto &B = F.getArg<2>();

    F.beginFunction();
    {
      auto I = F.defVar<int>(0, "i");
      auto J = F.defVar<int>(0, "j");
      auto K = F.defVar<int>(0, "k");
      auto UbnI = F.defRuntimeConst<int>(N, "ubn_i");
      auto UbnJ = F.defRuntimeConst<int>(N, "ubn_j");
      auto UbnK = F.defRuntimeConst<int>(N, "ubn_k");
      auto IncOne = F.defRuntimeConst<int>(1, "inc");
      auto Zero = F.defRuntimeConst<int>(0, "zero");

      F.buildLoopNest(F.forLoop<int>({I, Zero, UbnI, IncOne}).tile(TileI),
                      F.forLoop<int>({J, Zero, UbnJ, IncOne}).tile(TileJ),
                      F.forLoop<int>({K, Zero, UbnK, IncOne},
                                     [&]() {
                                       auto CIdx = I * N + J;
                                       auto AIdx = I * N + K;
                                       auto BIdx = K * N + J;
                                       C[CIdx] += A[AIdx] * B[BIdx];
                                     })
                          .tile(TileK))
          .emit();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main(int argc, char **argv) {
  proteus::init();

  if (argc != 3 && argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <N> <Tile> | <N> <TileI> <TileJ> <TileK>\n";
    proteus::finalize();
    return 1;
  }

  int N = std::atoi(argv[1]);
  int TileI = 0, TileJ = 0, TileK = 0;
  if (argc == 3) {
    int Tile = std::atoi(argv[2]);
    TileI = TileJ = TileK = Tile;
  } else {
    TileI = std::atoi(argv[2]);
    TileJ = std::atoi(argv[3]);
    TileK = std::atoi(argv[4]);
  }

  auto [JitMod, F] = getTiledMatmulFunction(N, TileI, TileJ, TileK);

  JitMod->compile();

  double *A = (double *)new double[N * N];
  double *B = (double *)new double[N * N];
  double *C = (double *)new double[N * N];

  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      A[I * N + J] = (1.0);
      B[I * N + J] = (1.0);
      C[I * N + J] = 0.0;
    }
  }

  F(C, A, B);

  bool Success = true;
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if (C[I * N + J] != N) {
        std::cout << "C[" << I << "][" << J << "] = " << C[I * N + J] << '\n';
        Success = false;
        break;
      }
    }
  }

  if (Success) {
    std::cout << "Verification successful\n";
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Verification successful
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
