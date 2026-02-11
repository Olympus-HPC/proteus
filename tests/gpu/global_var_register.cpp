// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CACHE_DIR="%t.$$.proteus" %build/global_var_register.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.h>

struct LargeStruct {
  char Blob[107];
  int Val;
};

struct SmallStruct {
  char Blob[3];
};

__device__ char CharVar;
__device__ int IntVar;
__device__ long LongVar;
__device__ float FloatVar;
__device__ double DoubleVar;
__device__ LargeStruct LSVar;
__device__ SmallStruct SSVar;

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("CharVar has size of %ld\n", sizeof(CharVar));
  printf("IntVar has size of %ld\n", sizeof(IntVar));
  printf("LongVar has size of %ld\n", sizeof(LongVar));
  printf("FloatVar has size of %ld\n", sizeof(FloatVar));
  printf("DoubleVar has size of %ld\n", sizeof(DoubleVar));
  printf("LSVar has size of %ld\n", sizeof(LSVar));
  printf("SSVar has size of %ld\n", sizeof(SSVar));
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CharVar
// CHECK-DAG: [GVarInfo]: CharVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_CHAR:[0-9]+]]
// CHECK-DAG: CharVar has size of [[SZ_CHAR]]

// IntVar
// CHECK-DAG: [GVarInfo]: IntVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_INT:[0-9]+]]
// CHECK-DAG: IntVar has size of [[SZ_INT]]

// LongVar
// CHECK-DAG: [GVarInfo]: LongVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LONG:[0-9]+]]
// CHECK-DAG: LongVar has size of [[SZ_LONG]]

// FloatVar
// CHECK-DAG: [GVarInfo]: FloatVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_FLOAT:[0-9]+]]
// CHECK-DAG: FloatVar has size of [[SZ_FLOAT]]

// DoubleVar
// CHECK-DAG: [GVarInfo]: DoubleVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_DOUBLE:[0-9]+]]
// CHECK-DAG: DoubleVar has size of [[SZ_DOUBLE]]

// LSVar
// CHECK-DAG: [GVarInfo]: LSVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LS:[0-9]+]]
// CHECK-DAG: LSVar has size of [[SZ_LS]]

// SSVar
// CHECK-DAG: [GVarInfo]: SSVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_SS:[0-9]+]]
// CHECK-DAG: SSVar has size of [[SZ_SS]]
