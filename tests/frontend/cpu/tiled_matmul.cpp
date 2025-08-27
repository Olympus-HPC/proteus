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

    auto &I = F.declVar<int>("i");
    auto &J = F.declVar<int>("j");
    auto &K = F.declVar<int>("k");
    auto &IncOne = F.declVar<int>("inc");
    auto &UbnI = F.declVar<int>("ubn_i");
    auto &UbnJ = F.declVar<int>("ubn_j");
    auto &UbnK = F.declVar<int>("ubn_k");

    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    F.beginFunction();
    {
      I = 0;
      J = 0;
      K = 0;
      UbnI = N;
      UbnJ = N;
      UbnK = N;
      IncOne = 1;
      auto &Zero = F.declVar<int>("zero");
      Zero = 0;

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

  //   for (int i = 0; i < N; i++) {
  //     for (int j = 0; j < N; j++) {
  //       A[i*N+j] = (1.0);
  //       B[i*N+j] = (1.0);
  //       C[i*N+j] = 0.0;
  //     }
  //   }

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
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     std::cout << C[i*N + j] << (j + 1 == N ? '\n' : ' ');
  //   }
  // }

  proteus::finalize();
  return 0;
}

// clang-format off
// No FileCheck for now; program prints the resulting C matrix

