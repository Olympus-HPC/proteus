// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_gpu_index_lowering.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_gpu_index_lowering.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/JitFrontend.h>

#include "../../../gpu/gpu_common.h"

#include <iostream>

using namespace proteus;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET, "mlir");
  auto KernelHandle = J.addKernel<void(unsigned int *)>("index_lowering");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &Out = F.getArg<0>();

    auto Scratch = F.declVar<size_t[]>(4, AddressSpace::DEFAULT, "scratch");
    auto I = F.declVar<size_t>("i");
    auto Zero = F.defVar<size_t>(0);
    auto Four = F.defVar<size_t>(4);
    auto One = F.defVar<size_t>(1);
    auto Base = F.defVar<size_t>(7);

    // Drive scf/index lowering with a size_t induction variable and use the
    // same values as memref indices in the lowered kernel body.
    F.beginFor(I, Zero, Four, One);
    {
      Scratch[I] = I * I + Base;
    }
    F.endFor();

    F.beginFor(I, Zero, Four, One);
    {
      Out[I] = F.convert<unsigned int>(Scratch[I] + I);
    }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  unsigned int *Out;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(unsigned int) * 4));
  for (size_t I = 0; I < 4; ++I)
    Out[I] = 0;

  J.compile();

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Out));
  gpuErrCheck(gpuDeviceSynchronize());

  bool Verified = true;
  constexpr size_t Expected[4] = {7, 9, 13, 19};
  for (size_t I = 0; I < 4; ++I) {
    std::cout << "Out[" << I << "] = " << Out[I] << "\n";
    if (Out[I] != Expected[I])
      Verified = false;
  }

  gpuErrCheck(gpuFree(Out));
  return Verified ? 0 : 1;
}

// clang-format off
// CHECK: Out[0] = 7
// CHECK-NEXT: Out[1] = 9
// CHECK-NEXT: Out[2] = 13
// CHECK-NEXT: Out[3] = 19
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
