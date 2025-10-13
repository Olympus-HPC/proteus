// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/atomics.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <limits>

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

  auto KernelHandle =
      J.addKernelTT<void(int *, float *, int)>("atomic_ops_kernel");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto [IntCounters, FloatCounters, N] = F.getArgsTT();

    auto Tid = F.callBuiltin(getThreadIdX);
    auto Bid = F.callBuiltin(getBlockIdX);
    auto BlockDim = F.callBuiltin(getBlockDimX);

    auto I = F.declVarTT<int>();
    I = Bid * BlockDim + Tid;

    auto IntAdd = IntCounters + 0;
    auto IntSub = IntCounters + 1;
    auto IntMax = IntCounters + 2;
    auto IntMin = IntCounters + 3;

    auto FloatAdd = FloatCounters + 0;
    auto FloatSub = FloatCounters + 1;
    auto FloatMax = FloatCounters + 2;
    auto FloatMin = FloatCounters + 3;

    auto IntOne = F.defVarTT<int>(1);
    auto FloatHalf = F.defVarTT<float>(0.5f);
    auto FloatOne = F.defVarTT<float>(1.0f);

    F.beginIfTT(I < N);
    {
      F.atomicAdd(IntAdd, IntOne);
      F.atomicSub(IntSub, IntOne);
      F.atomicMax(IntMax, I);
      F.atomicMin(IntMin, I);

      F.atomicAdd(FloatAdd, FloatHalf);
      F.atomicSub(FloatSub, FloatOne);

      auto IdxAsFloat = F.convertTT<float>(I);
      F.atomicMax(FloatMax, IdxAsFloat);
      F.atomicMin(FloatMin, IdxAsFloat);
    }
    F.endIfTT();

    F.ret();
  }
  F.endFunction();

  constexpr int N = 1024;
  int *IntCounters;
  float *FloatCounters;
  gpuErrCheck(gpuMallocManaged(&IntCounters, 4 * sizeof(int)));
  gpuErrCheck(gpuMallocManaged(&FloatCounters, 4 * sizeof(float)));

  IntCounters[0] = 0;
  IntCounters[1] = N;
  IntCounters[2] = std::numeric_limits<int>::lowest();
  IntCounters[3] = N;

  FloatCounters[0] = 0.0f;
  FloatCounters[1] = static_cast<float>(N);
  FloatCounters[2] = -std::numeric_limits<float>::infinity();
  FloatCounters[3] = std::numeric_limits<float>::infinity();

  J.print();
  J.compile();

  constexpr unsigned ThreadsPerBlock = 256;
  unsigned NumBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

  std::cout << "Launching with NumBlocks " << NumBlocks << " each with "
            << ThreadsPerBlock << " threads..." << '\n';
  gpuErrCheck(KernelHandle.launch({NumBlocks, 1, 1}, {ThreadsPerBlock, 1, 1}, 0,
                                  nullptr, IntCounters, FloatCounters, N));

  gpuErrCheck(gpuDeviceSynchronize());

  bool IntVerified = (IntCounters[0] == N) && (IntCounters[1] == 0) &&
                     (IntCounters[2] == N - 1) && (IntCounters[3] == 0);
  bool FloatVerified = (FloatCounters[0] == N * 0.5f) &&
                       (FloatCounters[1] == 0.0f) &&
                       (FloatCounters[2] == static_cast<float>(N - 1)) &&
                       (FloatCounters[3] == 0.0f);

  if (IntVerified) {
    std::cout << "Integer atomic ops verified!" << '\n';
  } else {
    std::cout << "Integer atomic ops failed!" << '\n';
  }

  if (FloatVerified) {
    std::cout << "Float atomic ops verified!" << '\n';
  } else {
    std::cout << "Float atomic ops failed!" << '\n';
  }

  gpuErrCheck(gpuFree(IntCounters));
  gpuErrCheck(gpuFree(FloatCounters));
  return (IntVerified && FloatVerified) ? 0 : 1;
}

// clang-format off
// CHECK: atomicrmw add
// CHECK: atomicrmw sub
// CHECK: atomicrmw max
// CHECK: atomicrmw min
// CHECK: atomicrmw fadd
// CHECK: atomicrmw fsub
// CHECK: atomicrmw fmax
// CHECK: atomicrmw fmin
// CHECK: Integer atomic ops verified!
// CHECK: Float atomic ops verified!
