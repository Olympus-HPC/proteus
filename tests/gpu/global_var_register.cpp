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

__device__ char char_var;
__device__ int int_var;
__device__ long long_var;
__device__ float float_var;
__device__ double double_var;
__device__ LargeStruct ls_var;
__device__ SmallStruct ss_var;

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("char_var has size of %ld\n", sizeof(char_var));
  printf("int_var has size of %ld\n", sizeof(int_var));
  printf("long_var has size of %ld\n", sizeof(long_var));
  printf("float_var has size of %ld\n", sizeof(float_var));
  printf("double_var has size of %ld\n", sizeof(double_var));
  printf("ls_var has size of %ld\n", sizeof(ls_var));
  printf("ss_var has size of %ld\n", sizeof(ss_var));
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// char_var
// CHECK-DAG: [GVarInfo]: char_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_CHAR:[0-9]+]]
// CHECK-DAG: char_var has size of [[SZ_CHAR]]

// int_var
// CHECK-DAG: [GVarInfo]: int_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_INT:[0-9]+]]
// CHECK-DAG: int_var has size of [[SZ_INT]]

// long_var
// CHECK-DAG: [GVarInfo]: long_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LONG:[0-9]+]] 
// CHECK-DAG: long_var has size of [[SZ_LONG]]

// float_var
// CHECK-DAG: [GVarInfo]: float_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_FLOAT:[0-9]+]] 
// CHECK-DAG: float_var has size of [[SZ_FLOAT]]

// double_var
// CHECK-DAG: [GVarInfo]: double_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_DOUBLE:[0-9]+]] 
// CHECK-DAG: double_var has size of [[SZ_DOUBLE]]

// ls_var
// CHECK-DAG: [GVarInfo]: ls_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_LS:[0-9]+]] 
// CHECK-DAG: ls_var has size of [[SZ_LS]]

// ss_var
// CHECK-DAG: [GVarInfo]: ss_var HAddr:{{0x[0-9a-f]+}} DevAddr:{{0x[0-9a-f]+}} VarSize:[[SZ_SS:[0-9]+]] 
// CHECK-DAG: ss_var has size of [[SZ_SS]]
