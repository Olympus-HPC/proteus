// RUN: rm -rf .proteus
// RUN: ./tiled_matmul
// RUN: rm -rf .proteus

#include <chrono>
#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto getTiledMatmulFunction(int N, int TileSize) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F =
      JitMod->addFunction<void, double *, double *, double *>("tiled_matmul");
  {

    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    F.beginFunction();
    {
      auto &I = F.defVar<int>(0, "i");
      auto &J = F.defVar<int>(0, "j");
      auto &K = F.defVar<int>(0, "k");
      auto &UbnI = F.defRuntimeConst(N, "ubn_i");
      auto &UbnJ = F.defRuntimeConst(N, "ubn_j");
      auto &UbnK = F.defRuntimeConst(N, "ubn_k");
      auto &IncOne = F.defRuntimeConst(1, "inc");
      auto &Zero = F.defRuntimeConst(0, "zero");

      F.buildLoopNest(
           {F.transformableForLoop({I, Zero, UbnI, IncOne}).tile(TileSize),
            F.transformableForLoop({J, Zero, UbnJ, IncOne}).tile(TileSize),
            F.transformableForLoop({K, Zero, UbnK, IncOne},
                                   [&]() {
                                     auto CIdx = I * N + J;
                                     auto AIdx = I * N + K;
                                     auto BIdx = K * N + J;
                                     C[CIdx] += A[AIdx] * B[BIdx];
                                   })
                .tile(TileSize)})
          .emit();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int N = 1024;
  constexpr int TileSize = 4;
  auto [JitMod, F] = getTiledMatmulFunction(N, TileSize);

  JitMod->compile();

  double *A = (double *)new double[N * N];
  double *B = (double *)new double[N * N];
  double *C = (double *)new double[N * N];

    for (int I = 0; I < N; I++) {
      for (int J = 0; J < N; J++) {
        A[I*N+J] = (1.0);
        B[I*N+J] = (1.0);
        C[I*N+J] = 0.0;
      }
    }

  F(C, A, B);

  // Timed trials
  const int NumTrials = 5;
  double TotalMs = 0.0;
  for (int T = 0; T < NumTrials; ++T) {
    auto Start = std::chrono::high_resolution_clock::now();
    F(C, A, B);
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs += Ms.count();
  }

  double AvgMs = TotalMs / static_cast<double>(NumTrials);
  std::cerr.setf(std::ios::fixed);
  std::cerr.precision(3);
  std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms"
            << '\n';

  // Print a small subset to avoid excessive output
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if(C[I*N + J] != 6*N) {
        std::cout << "C[" << I << "][" << J << "] = " << C[I*N + J] << '\n';
        break;
      }
    }
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// No FileCheck for now; program prints the resulting C matrix

