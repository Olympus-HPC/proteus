// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/add_vectors.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/add_vectors.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

using namespace proteus;
using namespace builtins::gpu;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET);

  // Add a kernel with the signature: void add_vectors(double *A, double *B,
  // size_t N)
  auto KernelHandle =
      J.addKernel<void(double *, double *, size_t)>("add_vectors");
  auto &F = KernelHandle.F;

  // Begin the function body.
  F.beginFunction();
  {
    // Declare local variables and argument getters.
    auto I = F.declVar<size_t>("I");
    auto Inc = F.declVar<size_t>("Inc");
    auto &A = F.getArg<0>(); // Pointer to vector A
    auto &B = F.getArg<1>(); // Pointer to vector B
    auto &N = F.getArg<2>(); // Vector size

    // Compute the global thread index.
    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);

    // Compute the stride (accesses number of threads).
    Inc = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    // Strided loop: each thread processes multiple elements.
    F.beginFor(I, I, N, Inc);
    { A[I] = A[I] + B[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  // Allocate and initialize input vectors A and B, and specify their size N.
  double *A;       // Pointer to vector A
  double *B;       // Pointer to vector B
  size_t N = 1024; // Number of elements in each vector
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&B, sizeof(double) * N));
  for (size_t I = 0; I < N; ++I) {
    A[I] = 1.0;
    B[I] = 2.0;
  }

  J.print();
  // Finalize and compile the JIT module. No further code can be added after
  // this.
  J.compile();

  // Configure the CUDA kernel launch parameters.
  constexpr unsigned ThreadsPerBlock = 256;
  unsigned NumBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

  // Launch the JIT-compiled kernel with the specified grid and block
  // dimensions. Arguments: grid size, block size, shared memory size (0),
  // stream (nullptr), kernel arguments (A, B, N).
  std::cout << "Launching with NumBlocks " << NumBlocks << " each with "
            << ThreadsPerBlock << " threads...\n";
  gpuErrCheck(KernelHandle.launch({NumBlocks, 1, 1}, {ThreadsPerBlock, 1, 1}, 0,
                                  nullptr, A, B, N));

  // Synchronize to ensure kernel execution is complete.
  gpuErrCheck(gpuDeviceSynchronize());

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

  gpuErrCheck(gpuFree(A));
  gpuErrCheck(gpuFree(B));

  return 0;
}

// clang-format off
// CHECK: fadd contract
// CHECK: Verification successful!
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache procuid 0 hits 1 accesses 1
