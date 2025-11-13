// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/scale100/scale100.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/scale100/scale100.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "../gpu_common.h"
#include <cstdio>
#include <cstdlib>

#include <proteus/JitInterface.hpp>

__global__ void foo0(int *, int *, int);
__global__ void foo1(int *, int *, int);
__global__ void foo2(int *, int *, int);
__global__ void foo3(int *, int *, int);
__global__ void foo4(int *, int *, int);
__global__ void foo5(int *, int *, int);
__global__ void foo6(int *, int *, int);
__global__ void foo7(int *, int *, int);
__global__ void foo8(int *, int *, int);
__global__ void foo9(int *, int *, int);
__global__ void foo10(int *, int *, int);
__global__ void foo11(int *, int *, int);
__global__ void foo12(int *, int *, int);
__global__ void foo13(int *, int *, int);
__global__ void foo14(int *, int *, int);
__global__ void foo15(int *, int *, int);
__global__ void foo16(int *, int *, int);
__global__ void foo17(int *, int *, int);
__global__ void foo18(int *, int *, int);
__global__ void foo19(int *, int *, int);
__global__ void foo20(int *, int *, int);
__global__ void foo21(int *, int *, int);
__global__ void foo22(int *, int *, int);
__global__ void foo23(int *, int *, int);
__global__ void foo24(int *, int *, int);
__global__ void foo25(int *, int *, int);
__global__ void foo26(int *, int *, int);
__global__ void foo27(int *, int *, int);
__global__ void foo28(int *, int *, int);
__global__ void foo29(int *, int *, int);
__global__ void foo30(int *, int *, int);
__global__ void foo31(int *, int *, int);
__global__ void foo32(int *, int *, int);
__global__ void foo33(int *, int *, int);
__global__ void foo34(int *, int *, int);
__global__ void foo35(int *, int *, int);
__global__ void foo36(int *, int *, int);
__global__ void foo37(int *, int *, int);
__global__ void foo38(int *, int *, int);
__global__ void foo39(int *, int *, int);
__global__ void foo40(int *, int *, int);
__global__ void foo41(int *, int *, int);
__global__ void foo42(int *, int *, int);
__global__ void foo43(int *, int *, int);
__global__ void foo44(int *, int *, int);
__global__ void foo45(int *, int *, int);
__global__ void foo46(int *, int *, int);
__global__ void foo47(int *, int *, int);
__global__ void foo48(int *, int *, int);
__global__ void foo49(int *, int *, int);
__global__ void foo50(int *, int *, int);
__global__ void foo51(int *, int *, int);
__global__ void foo52(int *, int *, int);
__global__ void foo53(int *, int *, int);
__global__ void foo54(int *, int *, int);
__global__ void foo55(int *, int *, int);
__global__ void foo56(int *, int *, int);
__global__ void foo57(int *, int *, int);
__global__ void foo58(int *, int *, int);
__global__ void foo59(int *, int *, int);
__global__ void foo60(int *, int *, int);
__global__ void foo61(int *, int *, int);
__global__ void foo62(int *, int *, int);
__global__ void foo63(int *, int *, int);
__global__ void foo64(int *, int *, int);
__global__ void foo65(int *, int *, int);
__global__ void foo66(int *, int *, int);
__global__ void foo67(int *, int *, int);
__global__ void foo68(int *, int *, int);
__global__ void foo69(int *, int *, int);
__global__ void foo70(int *, int *, int);
__global__ void foo71(int *, int *, int);
__global__ void foo72(int *, int *, int);
__global__ void foo73(int *, int *, int);
__global__ void foo74(int *, int *, int);
__global__ void foo75(int *, int *, int);
__global__ void foo76(int *, int *, int);
__global__ void foo77(int *, int *, int);
__global__ void foo78(int *, int *, int);
__global__ void foo79(int *, int *, int);
__global__ void foo80(int *, int *, int);
__global__ void foo81(int *, int *, int);
__global__ void foo82(int *, int *, int);
__global__ void foo83(int *, int *, int);
__global__ void foo84(int *, int *, int);
__global__ void foo85(int *, int *, int);
__global__ void foo86(int *, int *, int);
__global__ void foo87(int *, int *, int);
__global__ void foo88(int *, int *, int);
__global__ void foo89(int *, int *, int);
__global__ void foo90(int *, int *, int);
__global__ void foo91(int *, int *, int);
__global__ void foo92(int *, int *, int);
__global__ void foo93(int *, int *, int);
__global__ void foo94(int *, int *, int);
__global__ void foo95(int *, int *, int);
__global__ void foo96(int *, int *, int);
__global__ void foo97(int *, int *, int);
__global__ void foo98(int *, int *, int);
__global__ void foo99(int *, int *, int);

int main() {
  proteus::init();

  int *A = nullptr;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(int) * 100));
  int *B = nullptr;
  gpuErrCheck(gpuMallocManaged(&B, sizeof(int) * 100));
  for (int I = 0; I < 100; ++I) {
    A[I] = 0;
    B[I] = 1;
  }

  int NumBlocks = std::max(1, 100 / 256);

  foo0<<<NumBlocks, 256>>>(A, B, 100);
  foo1<<<NumBlocks, 256>>>(A, B, 100);
  foo2<<<NumBlocks, 256>>>(A, B, 100);
  foo3<<<NumBlocks, 256>>>(A, B, 100);
  foo4<<<NumBlocks, 256>>>(A, B, 100);
  foo5<<<NumBlocks, 256>>>(A, B, 100);
  foo6<<<NumBlocks, 256>>>(A, B, 100);
  foo7<<<NumBlocks, 256>>>(A, B, 100);
  foo8<<<NumBlocks, 256>>>(A, B, 100);
  foo9<<<NumBlocks, 256>>>(A, B, 100);
  foo10<<<NumBlocks, 256>>>(A, B, 100);
  foo11<<<NumBlocks, 256>>>(A, B, 100);
  foo12<<<NumBlocks, 256>>>(A, B, 100);
  foo13<<<NumBlocks, 256>>>(A, B, 100);
  foo14<<<NumBlocks, 256>>>(A, B, 100);
  foo15<<<NumBlocks, 256>>>(A, B, 100);
  foo16<<<NumBlocks, 256>>>(A, B, 100);
  foo17<<<NumBlocks, 256>>>(A, B, 100);
  foo18<<<NumBlocks, 256>>>(A, B, 100);
  foo19<<<NumBlocks, 256>>>(A, B, 100);
  foo20<<<NumBlocks, 256>>>(A, B, 100);
  foo21<<<NumBlocks, 256>>>(A, B, 100);
  foo22<<<NumBlocks, 256>>>(A, B, 100);
  foo23<<<NumBlocks, 256>>>(A, B, 100);
  foo24<<<NumBlocks, 256>>>(A, B, 100);
  foo25<<<NumBlocks, 256>>>(A, B, 100);
  foo26<<<NumBlocks, 256>>>(A, B, 100);
  foo27<<<NumBlocks, 256>>>(A, B, 100);
  foo28<<<NumBlocks, 256>>>(A, B, 100);
  foo29<<<NumBlocks, 256>>>(A, B, 100);
  foo30<<<NumBlocks, 256>>>(A, B, 100);
  foo31<<<NumBlocks, 256>>>(A, B, 100);
  foo32<<<NumBlocks, 256>>>(A, B, 100);
  foo33<<<NumBlocks, 256>>>(A, B, 100);
  foo34<<<NumBlocks, 256>>>(A, B, 100);
  foo35<<<NumBlocks, 256>>>(A, B, 100);
  foo36<<<NumBlocks, 256>>>(A, B, 100);
  foo37<<<NumBlocks, 256>>>(A, B, 100);
  foo38<<<NumBlocks, 256>>>(A, B, 100);
  foo39<<<NumBlocks, 256>>>(A, B, 100);
  foo40<<<NumBlocks, 256>>>(A, B, 100);
  foo41<<<NumBlocks, 256>>>(A, B, 100);
  foo42<<<NumBlocks, 256>>>(A, B, 100);
  foo43<<<NumBlocks, 256>>>(A, B, 100);
  foo44<<<NumBlocks, 256>>>(A, B, 100);
  foo45<<<NumBlocks, 256>>>(A, B, 100);
  foo46<<<NumBlocks, 256>>>(A, B, 100);
  foo47<<<NumBlocks, 256>>>(A, B, 100);
  foo48<<<NumBlocks, 256>>>(A, B, 100);
  foo49<<<NumBlocks, 256>>>(A, B, 100);
  foo50<<<NumBlocks, 256>>>(A, B, 100);
  foo51<<<NumBlocks, 256>>>(A, B, 100);
  foo52<<<NumBlocks, 256>>>(A, B, 100);
  foo53<<<NumBlocks, 256>>>(A, B, 100);
  foo54<<<NumBlocks, 256>>>(A, B, 100);
  foo55<<<NumBlocks, 256>>>(A, B, 100);
  foo56<<<NumBlocks, 256>>>(A, B, 100);
  foo57<<<NumBlocks, 256>>>(A, B, 100);
  foo58<<<NumBlocks, 256>>>(A, B, 100);
  foo59<<<NumBlocks, 256>>>(A, B, 100);
  foo60<<<NumBlocks, 256>>>(A, B, 100);
  foo61<<<NumBlocks, 256>>>(A, B, 100);
  foo62<<<NumBlocks, 256>>>(A, B, 100);
  foo63<<<NumBlocks, 256>>>(A, B, 100);
  foo64<<<NumBlocks, 256>>>(A, B, 100);
  foo65<<<NumBlocks, 256>>>(A, B, 100);
  foo66<<<NumBlocks, 256>>>(A, B, 100);
  foo67<<<NumBlocks, 256>>>(A, B, 100);
  foo68<<<NumBlocks, 256>>>(A, B, 100);
  foo69<<<NumBlocks, 256>>>(A, B, 100);
  foo70<<<NumBlocks, 256>>>(A, B, 100);
  foo71<<<NumBlocks, 256>>>(A, B, 100);
  foo72<<<NumBlocks, 256>>>(A, B, 100);
  foo73<<<NumBlocks, 256>>>(A, B, 100);
  foo74<<<NumBlocks, 256>>>(A, B, 100);
  foo75<<<NumBlocks, 256>>>(A, B, 100);
  foo76<<<NumBlocks, 256>>>(A, B, 100);
  foo77<<<NumBlocks, 256>>>(A, B, 100);
  foo78<<<NumBlocks, 256>>>(A, B, 100);
  foo79<<<NumBlocks, 256>>>(A, B, 100);
  foo80<<<NumBlocks, 256>>>(A, B, 100);
  foo81<<<NumBlocks, 256>>>(A, B, 100);
  foo82<<<NumBlocks, 256>>>(A, B, 100);
  foo83<<<NumBlocks, 256>>>(A, B, 100);
  foo84<<<NumBlocks, 256>>>(A, B, 100);
  foo85<<<NumBlocks, 256>>>(A, B, 100);
  foo86<<<NumBlocks, 256>>>(A, B, 100);
  foo87<<<NumBlocks, 256>>>(A, B, 100);
  foo88<<<NumBlocks, 256>>>(A, B, 100);
  foo89<<<NumBlocks, 256>>>(A, B, 100);
  foo90<<<NumBlocks, 256>>>(A, B, 100);
  foo91<<<NumBlocks, 256>>>(A, B, 100);
  foo92<<<NumBlocks, 256>>>(A, B, 100);
  foo93<<<NumBlocks, 256>>>(A, B, 100);
  foo94<<<NumBlocks, 256>>>(A, B, 100);
  foo95<<<NumBlocks, 256>>>(A, B, 100);
  foo96<<<NumBlocks, 256>>>(A, B, 100);
  foo97<<<NumBlocks, 256>>>(A, B, 100);
  foo98<<<NumBlocks, 256>>>(A, B, 100);
  foo99<<<NumBlocks, 256>>>(A, B, 100);
  gpuErrCheck(gpuDeviceSynchronize());

  bool VecaddSuccess = true;
  for (int I = 0; I < 100; ++I)
    if (A[I] != 200) {
      VecaddSuccess = false;
      break;
    }

  if (VecaddSuccess)
    fprintf(stdout, "Verification successful\n");
  else {
    fprintf(stdout, "Vecadd failed\n");
    fprintf(stdout, "Verification failed\n");
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Verification successful
// CHECK-COUNT-100: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 100
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 100 accesses 100
