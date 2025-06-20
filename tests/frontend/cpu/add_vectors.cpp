// RUN: rm -rf .proteus
// RUN: ./add_vectors | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

int main() {
  auto J = proteus::JitModule("host");

  // Add a function with the signature:
  //  void add_vectors(double *A, double *B, size_t N)
  auto &F = J.addFunction<void, double *, double *, size_t>("add_vectors");

  // Begin the function body.
  F.beginFunction();
  {
    // Declare local variables and argument getters.
    auto &I = F.declVar<size_t>("I");
    auto &Inc = F.declVar<size_t>("Inc");
    auto &A = F.getArg(0); // Pointer to vector A
    auto &B = F.getArg(1); // Pointer to vector B
    auto &N = F.getArg(2); // Vector size

    // Element-wise addition over all vector elements.
    I = 0;
    Inc = 1;
    F.beginFor(I, I, N, Inc);
    { A[I] = A[I] + B[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  // Allocate and initialize input vectors A and B, and specify their size N.
  size_t N = 1024;           // Number of elements in each vector
  double *A = new double[N]; // Pointer to vector A
  double *B = new double[N]; // Pointer to vector B
  for (size_t I = 0; I < N; ++I) {
    A[I] = 1.0;
    B[I] = 2.0;
  }

  J.print();
  // Finalize and compile the JIT module. No further code can be added after
  // this.
  J.compile();

  // Run the function.
  J.run<void>(F, A, B, N);

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
