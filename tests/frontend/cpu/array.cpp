// RUN: rm -rf .proteus
// RUN: ./array | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <proteus/JitFrontend.hpp>

using namespace proteus;

int main() {
  auto J = proteus::JitModule("host");

  auto &F = J.addFunction<void, double *, double *>("arrays_test");

  F.beginFunction();
  {
    auto &OutLocal = F.getArg(0);
    auto &OutGlobal = F.getArg(1);

    auto &I = F.defVar<size_t>(0, "I");
    auto &Inc = F.defVar<size_t>(1, "Inc");

    auto &Local = F.declVar<double[]>(16, AddressSpace::DEFAULT, "local_array");
    auto &Global = F.declVar<double[]>(16, AddressSpace::GLOBAL, "global_array");

    auto &Bound = F.defRuntimeConst<size_t>(16, "Bound");

    F.beginFor(I, I, Bound, Inc);
    { Local[I] = 2.0 * I; }
    F.endFor();

    I = 0;
    F.beginFor(I, I, Bound, Inc);
    { Global[I] = 1000.0 + I; }
    F.endFor();

    I = 0;
    F.beginFor(I, I, Bound, Inc);
    { OutLocal[I] = Local[I]; }
    F.endFor();

    I = 0;
    F.beginFor(I, I, Bound, Inc);
    { OutGlobal[I] = Global[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  constexpr size_t N = 16;
  double *OutLocal = new double[N];
  double *OutGlobal = new double[N];

  J.print();
  J.compile();

  F(OutLocal, OutGlobal);

  bool Verified = true;
  for (size_t I = 0; I < N; ++I) {
    double ExpectedLocal = 2.0 * I;
    double ExpectedGlobal = 1000.0 + I;
    if (OutLocal[I] != ExpectedLocal) {
      std::cout << "Verification failed: OutLocal[" << I << "] = "
                << OutLocal[I] << " != " << ExpectedLocal << " (expected)\n";
      Verified = false;
      break;
    }
    if (OutGlobal[I] != ExpectedGlobal) {
      std::cout << "Verification failed: OutGlobal[" << I << "] = "
                << OutGlobal[I] << " != " << ExpectedGlobal << " (expected)\n";
      Verified = false;
      break;
    }
  }

  if (Verified)
    std::cout << "Verification successful!\n";

  delete[] OutLocal;
  delete[] OutGlobal;
  return Verified ? 0 : 1;
}

// clang-format off
// CHECK: @global_array = internal addrspace(1) global [{{[0-9]+}} x double] undef
// CHECK: %local_array = alloca [{{[0-9]+}} x double], align 8
// CHECK: Verification successful!

