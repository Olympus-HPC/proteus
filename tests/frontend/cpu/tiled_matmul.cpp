// RUN: rm -rf .proteus
// RUN: ./tiled_matmul | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto getTiledMatmulFunction(int N) {
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
           {F.forLoop({I, Zero, UbnI, IncOne}).tile(3),
            F.forLoop({J, Zero, UbnJ, IncOne}).tile(4),
            F.forLoop({K, Zero, UbnK, IncOne},
                                   [&]() {
                                     auto CIdx = I * N + J;
                                     auto AIdx = I * N + K;
                                     auto BIdx = K * N + J;
                                     C[CIdx] += A[AIdx] * B[BIdx];
                                   })
                .tile(5)})
          .emit();

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main() {
  proteus::init();
  constexpr int N = 16;
  auto [JitMod, F] = getTiledMatmulFunction(N);

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

  bool Success = true;
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if(C[I*N + J] != N) {
        std::cout << "C[" << I << "][" << J << "] = " << C[I*N + J] << '\n';
        Success = false;
        break;
      }
    }
  }

  if(Success) {
    std::cout << "Verification successful\n";
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Verification successful

