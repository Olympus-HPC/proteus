// RUN: rm -rf .proteus
// RUN: ./for | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <chrono>
#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

auto getTiledMatmulFunction(int N, int TILE_SIZE) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F =
      JitMod->addFunction<void, double *, double *, double *>("tiled_matmul");
  {

    auto &I = F.declVar<int>("i");
    auto &J = F.declVar<int>("j");
    auto &K = F.declVar<int>("k");
    auto &IncOne = F.declVar<int>("inc");
    auto &UBN_I = F.declVar<int>("ubn_i");
    auto &UBN_J = F.declVar<int>("ubn_j");
    auto &UBN_K = F.declVar<int>("ubn_k");

    auto args = F.getArgs();
    auto &C = std::get<0>(args);
    auto &A = std::get<1>(args);
    auto &B = std::get<2>(args);

    F.beginFunction();
    {
      I = 0;
      J = 0;
      K = 0;
      UBN_I = N;
      UBN_J = N;
      UBN_K = N;
      IncOne = 1;
      auto &Zero = F.declVar<int>("zero");
      Zero = 0;

      F.LoopNest({F.ForLoop({I, Zero, UBN_I, IncOne}).tile(TILE_SIZE),
                  F.ForLoop({J, Zero, UBN_J, IncOne}).tile(TILE_SIZE),
                  F.ForLoop({K, Zero, UBN_K, IncOne},
                            [&]() {
                              auto CIdx = I * N + J;
                              auto AIdx = I * N + K;
                              auto BIdx = K * N + J;
                              C[CIdx] += A[AIdx] * B[BIdx];
                            }).tile(TILE_SIZE)
        })
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
  constexpr int TILE_SIZE = 4;
  auto [JitMod, F] = getTiledMatmulFunction(N, TILE_SIZE);

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
  const int num_trials = 5;
  double total_ms = 0.0;
  for (int t = 0; t < num_trials; ++t) {
    auto start = std::chrono::high_resolution_clock::now();
    F(C, A, B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    total_ms += ms.count();
  }

  double avg_ms = total_ms / static_cast<double>(num_trials);
  std::cerr.setf(std::ios::fixed);
  std::cerr.precision(3);
  std::cerr << "Average over " << num_trials << " trials: " << avg_ms << " ms"
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

