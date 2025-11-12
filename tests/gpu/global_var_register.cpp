// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CACHE_DIR="%t.$$.proteus" %build/global_var_register.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.hpp>

struct LargeStruct {
  char Blob[107];
  int Val;
};

struct SmallStruct {
  char Blob[3];
};

__device__ char charVar;
__device__ int intVar;
__device__ long longVar;
__device__ float floatVar;
__device__ double doubleVar;
__device__ LargeStruct lsVar;
__device__ SmallStruct ssVar;

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("charVar has size of %ld\n", sizeof(charVar));
  printf("intVar has size of %ld\n", sizeof(intVar));
  printf("longVar has size of %ld\n", sizeof(longVar));
  printf("floatVar has size of %ld\n", sizeof(floatVar));
  printf("doubleVar has size of %ld\n", sizeof(doubleVar));
  printf("lsVar has size of %ld\n", sizeof(lsVar));
  printf("ssVar has size of %ld\n", sizeof(ssVar));
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// charVar
// CHECK-DAG: [GVarInfo]: charVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_CHAR:[0-9]+]]
// CHECK-DAG: charVar has size of [[SZ_CHAR]]

// intVar
// CHECK-DAG: [GVarInfo]: intVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_INT:[0-9]+]]
// CHECK-DAG: intVar has size of [[SZ_INT]]

// longVar
// CHECK-DAG: [GVarInfo]: longVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LONG:[0-9]+]] 
// CHECK-DAG: longVar has size of [[SZ_LONG]]

// floatVar
// CHECK-DAG: [GVarInfo]: floatVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_FLOAT:[0-9]+]] 
// CHECK-DAG: floatVar has size of [[SZ_FLOAT]]

// doubleVar
// CHECK-DAG: [GVarInfo]: doubleVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_DOUBLE:[0-9]+]] 
// CHECK-DAG: doubleVar has size of [[SZ_DOUBLE]]

// lsVar
// CHECK-DAG: [GVarInfo]: lsVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LS:[0-9]+]] 
// CHECK-DAG: lsVar has size of [[SZ_LS]]

// ssVar
// CHECK-DAG: [GVarInfo]: ssVar HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_SS:[0-9]+]] 
// CHECK-DAG: ssVar has size of [[SZ_SS]]
