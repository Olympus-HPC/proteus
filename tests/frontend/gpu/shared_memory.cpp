// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/shared_memory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/shared_memory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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
constexpr unsigned WarpSize = 64;
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
constexpr unsigned WarpSize = 32;
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle = J.addKernel<void(double *)>("shared_reverse_warp");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &A = F.getArg(0);

    auto &Tid = F.callBuiltin(getThreadIdX);
    auto &Bid = F.callBuiltin(getBlockIdX);

    auto &I = F.declVar<size_t>("I");
    I = Bid * WarpSize + Tid;

    auto &S = F.declVar<double[]>(WarpSize, AddressSpace::SHARED, "shared_mem");

    // Load from global into shared, then sync, then write reversed index.
    S[Tid] = A[I];
    F.callBuiltin(syncThreads);
    A[I] = S[WarpSize - 1 - Tid];

    F.ret();
  }
  F.endFunction();

  // Host-side data
  constexpr size_t N = 2048;
  double *A;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(double) * N));

  // Initialize with ascending values within each warp-aligned block.
  for (size_t I = 0; I < N; ++I)
    A[I] = static_cast<double>(I % WarpSize);

  J.print();
  J.compile();

  constexpr unsigned ThreadsPerBlock = WarpSize;
  static_assert(N % ThreadsPerBlock == 0, "N must be multiple of WarpSize");
  unsigned NumBlocks = N / ThreadsPerBlock;

  std::cout << "Launching with NumBlocks " << NumBlocks << " each with "
            << ThreadsPerBlock << " threads...\n";
  gpuErrCheck(KernelHandle.launch({NumBlocks, 1, 1}, {ThreadsPerBlock, 1, 1}, 0,
                                  nullptr, A));

  gpuErrCheck(gpuDeviceSynchronize());

  bool Verified = true;
  for (size_t B = 0; B < NumBlocks && Verified; ++B) {
    for (size_t T = 0; T < WarpSize; ++T) {
      size_t Idx = B * WarpSize + T;
      double Expected = static_cast<double>(WarpSize - 1 - T);
      if (A[Idx] != Expected) {
        std::cout << "Verification failed: A[" << Idx << "] = " << A[Idx]
                  << " != " << Expected << " (expected)\n";
        Verified = false;
        break;
      }
    }
  }

  if (Verified)
    std::cout << "Verification successful!\n";

  gpuErrCheck(gpuFree(A));
  return Verified ? 0 : 1;
}

// clang-format off
// CHECK: @shared_mem = internal addrspace(3) global [{{[0-9]+}} x double] undef
// CHECK: Verification successful!
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
