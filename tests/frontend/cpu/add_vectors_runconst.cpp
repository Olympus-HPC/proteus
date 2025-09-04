// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/add_vectors_runconst | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/add_vectors_runconst | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

auto createJitFunction(size_t N) {
  auto J = std::make_unique<JitModule>("host");

  // Add a function with the signature: void add_vectors(double *A, double *B)
  // using the vector size N as a runtime constant.
  auto &F = J->addFunction<void, double *, double *>("add_vectors");

  // Begin the function body.
  F.beginFunction();
  {
    // Pointers to vectors A, B in arguments.
    auto [A, B] = F.getArgs();
    // Declare local variables and argument getters.
    auto &I = F.defVar<size_t>(0, "I");
    auto &Inc = F.defVar<size_t>(1, "Inc");
    // Runtime constant vector size
    auto &RunConstN = F.defRuntimeConst(N);
    // Element-wise addition over all vector elements.
    F.beginFor(I, I, RunConstN, Inc);
    { A[I] = A[I] + B[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), std::ref(F));
}

int main() {
  // Allocate and initialize input vectors A and B, and specify their size N.
  size_t N = 1024;           // Number of elements in each vector
  double *A = new double[N]; // Pointer to vector A
  double *B = new double[N]; // Pointer to vector B
  for (size_t I = 0; I < N; ++I) {
    A[I] = 1.0;
    B[I] = 2.0;
  }

  auto [J, F] = createJitFunction(N);
  F(A, B);

  bool Verified = true;
  for (size_t I = 0; I < N; ++I) {
    if (A[I] != 3.0) {
      std::cout << "Verification failed: A[" << I << "] = " << A[I]
                << " != 3.0 (expected)\n";
      Verified = false;
      break;
    }
  }
  if (Verified)
    std::cout << "Verification successful!\n";

  delete[] A;
  delete[] B;

  return 0;
}

// clang-format off
// CHECK: Verification successful!
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
