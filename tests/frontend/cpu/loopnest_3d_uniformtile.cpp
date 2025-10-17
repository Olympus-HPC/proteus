// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_3d_uniformtile | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/loopnest_3d_uniformtile | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdlib>
#include <iostream>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

static auto get3DUniformTileFunction(int DI, int DJ, int DK, int Tile) {
  auto JitMod = std::make_unique<proteus::JitModule>("host");
  auto &F = JitMod->addFunction<void(double *)>("loopnest_3d_uniformtile");

  auto I = F.declVar<int>("i");
  auto J = F.declVar<int>("j");
  auto K = F.declVar<int>("k");
  auto IncOne = F.declVar<int>("inc");
  auto UBI = F.declVar<int>("ubi");
  auto UBJ = F.declVar<int>("ubj");
  auto UBK = F.declVar<int>("ubk");

  F.beginFunction();
  {
    auto &A = F.getArg<0>();

    I = 0;
    J = 0;
    K = 0;
    UBI = DI;
    UBJ = DJ;
    UBK = DK;
    IncOne = 1;
    auto Zero = F.declVar<int>("zero");
    Zero = 0;

    F.buildLoopNest(F.forLoop<int>({I, Zero, UBI, IncOne}),
                    F.forLoop<int>({J, Zero, UBJ, IncOne}),
                    F.forLoop<int>({K, Zero, UBK, IncOne},
                                   [&]() {
                                     auto Idx = I * DJ * DK + J * DK + K;
                                     A[Idx] = I * 10000 + J * 100 + K;
                                   }))
        .tile(Tile)
        .emit();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(JitMod), std::ref(F));
}

int main(int argc, char **argv) {
  proteus::init();

  int DI = 8, DJ = 8, DK = 2;
  int Tile = 4;

  if (argc == 4) {
    DI = std::atoi(argv[1]);
    DJ = std::atoi(argv[2]);
    DK = std::atoi(argv[3]);
  } else if (argc == 5) {
    DI = std::atoi(argv[1]);
    DJ = std::atoi(argv[2]);
    DK = std::atoi(argv[3]);
    Tile = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [DI DJ DK [Tile]]\n";
    proteus::finalize();
    return 1;
  }

  auto [JitMod, F] = get3DUniformTileFunction(DI, DJ, DK, Tile);

  JitMod->compile();

  double *A = new double[DI * DJ * DK];
  for (int OuterI = 0; OuterI < DI * DJ * DK; OuterI++)
    A[OuterI] = 0.0;

  F(A);

  bool IsOk = true;
  for (int OuterI = 0; OuterI < DI; OuterI++) {
    for (int InnerJ = 0; InnerJ < DJ; InnerJ++) {
      for (int InnerK = 0; InnerK < DK; InnerK++) {
        double Expected = OuterI * 10000 + InnerJ * 100 + InnerK;
        if (A[OuterI * DJ * DK + InnerJ * DK + InnerK] != Expected) {
          std::cout << "Mismatch at (" << OuterI << "," << InnerJ << ","
                    << InnerK << ") got "
                    << A[OuterI * DJ * DK + InnerJ * DK + InnerK]
                    << " expected " << Expected << "\n";
          IsOk = false;
          break;
        }
      }
      if (!IsOk)
        break;
    }
    if (!IsOk)
      break;
  }

  if (IsOk) {
    std::cout << "Verification successful\n";
  }

  delete[] A;
  proteus::finalize();
  return 0;
}

// CHECK: Verification successful
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
