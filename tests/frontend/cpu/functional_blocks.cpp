// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/functional_blocks | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/functional_blocks | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.h>

constexpr auto Eager = proteus::EmissionPolicy::Eager;

int main() {
  auto J = proteus::JitModule();
  auto &F = J.addFunction<void(double *)>("functional_blocks");

  auto I = F.declVar<int>("i");
  auto UB = F.declVar<int>("ub");
  auto One = F.declVar<int>("one");
  auto Zero = F.declVar<int>("zero");
  auto &Arg = F.getArg<0>();

  F.function([&]() {
    UB = 10;
    One = 1;
    Zero = 0;

    I = Zero;
    F.forLoop<Eager>(I, I, UB, One, [&]() {
      F.ifThen(I % 2 == 0, [&]() { Arg[I] = Arg[I] + 2.0; });
    });

    I = Zero;
    F.whileLoop([&]() { return I < UB; },
                [&]() {
                  Arg[I] = Arg[I] + 1.0;
                  I = I + One;
                });

    F.ret();
  });

  J.compile();

  double X[10];
  for (int I = 0; I < 10; I++)
    X[I] = 1.0;

  F(X);
  for (int I = 0; I < 10; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  return 0;
}

// clang-format off
// CHECK: X[0] = 4
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 4
// CHECK-NEXT: X[3] = 2
// CHECK-NEXT: X[4] = 4
// CHECK-NEXT: X[5] = 2
// CHECK-NEXT: X[6] = 4
// CHECK-NEXT: X[7] = 2
// CHECK-NEXT: X[8] = 4
// CHECK-NEXT: X[9] = 2
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
