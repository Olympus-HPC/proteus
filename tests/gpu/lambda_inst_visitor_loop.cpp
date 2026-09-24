// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_inst_visitor_loop.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
overwriteSlot(F *volatile *Slot, F *Replacement) {
  *Slot = Replacement;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnSlotOnce(F *volatile *Slot) {
  return Slot;
}

// LLVM shape: a loop-carried pointer phi feeds a pointer-preserving helper and
// then feeds itself on the backedge. The only concrete origin is Slot, but a
// recursive provenance walk must distinguish that fixed point from ambiguity.
template <typename F>
__device__ __attribute__((noinline)) static F *volatile *
returnSlotThroughLoop(F *volatile *Slot, int Iterations) {
  F *volatile *Alias = Slot;
  for (int I = 0; I < Iterations; ++I)
    Alias = returnSlotOnce(Alias);
  return Alias;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughLoopSlot(F *Initial, F *Replacement, int Iterations) {
  F *volatile Slot = Initial;
  F *volatile *Alias = returnSlotThroughLoop(&Slot, Iterations);
  overwriteSlot(Alias, Replacement);
  (*Slot)();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelLoopSlot(F Initial, F Replacement, int Iterations) {
  invokeThroughLoopSlot(&Initial, &Replacement, Iterations);
}

// The backedge changes the pointer offset. The returned slot depends on the
// runtime trip count, so this cycle has no single static provenance result.
template <typename F>
__device__ __attribute__((noinline, optnone)) static F *volatile *
returnNextSlot(F *volatile *Slot) {
  return Slot + 1;
}

template <typename F>
__device__ __attribute__((noinline)) static F *volatile *
returnAdvancedSlotThroughLoop(F *volatile *Slot, int Iterations) {
  F *volatile *Alias = Slot;
  for (int I = 0; I < Iterations; ++I)
    Alias = returnNextSlot(Alias);
  return Alias;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) static void
invokeThroughOffsetLoop(F *Initial, F *Other, F *Replacement, int Iterations) {
  F *volatile Slots[2] = {Initial, Other};
  F *volatile *Alias = returnAdvancedSlotThroughLoop(Slots, Iterations);
  overwriteSlot(Alias, Replacement);
  (*Slots[1])();
}

template <typename F>
__global__ __attribute__((annotate("jit"))) static void
kernelOffsetLoop(F Initial, F Other, F Replacement, int Iterations) {
  invokeThroughOffsetLoop(&Initial, &Other, &Replacement, Iterations);
}

static auto makeLoopBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("loop slot %d\n", X);
      });
}

static auto makeOffsetLoopBody(int Value) {
  return proteus::register_lambda(
      [X = proteus::jit_variable(Value)] __host__ __device__ {
        printf("offset loop %d\n", X);
      });
}

int main() {
  auto Initial = makeLoopBody(641);
  auto Replacement = makeLoopBody(643);
  kernelLoopSlot<<<1, 1>>>(Initial, Replacement, 3);
  gpuErrCheck(gpuDeviceSynchronize());

  auto OffsetInitial = makeOffsetLoopBody(647);
  auto OffsetOther = makeOffsetLoopBody(649);
  auto OffsetReplacement = makeOffsetLoopBody(653);
  kernelOffsetLoop<<<1, 1>>>(OffsetInitial, OffsetOther, OffsetReplacement, 1);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: [LambdaSpec] Replacing slot 0 with i32 643
// CHECK: loop slot 643
// CHECK: [KernelConfig] ID:{{.*}}kernelOffsetLoop
// CHECK-NOT: [LambdaSpec]
// CHECK: offset loop 653
