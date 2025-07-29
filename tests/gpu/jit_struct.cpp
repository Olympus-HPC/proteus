// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./jit_struct.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./jit_struct.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

#include <proteus/JitInterface.hpp>

struct DimInt {
  int X, Y, Z;
};

struct DimDouble {
  double X, Y, Z;
};

struct DimFloat {
  float X, Y, Z;
};

struct DimMix {
  int X;
  double Y;
  float Z;
};

struct DimFill {
  int X, Y, Z;
  int Fill[10];
};

template <typename DimT> __global__ void testByPtr(DimT *A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A->X),
         static_cast<double>(A->Y), static_cast<double>(A->Z));
}
template <typename DimT> __global__ void testByRef(DimT &A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A.X),
         static_cast<double>(A.Y), static_cast<double>(A.Z));
}

template <typename DimT> __global__ void testByVal(DimT A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A.X),
         static_cast<double>(A.Y), static_cast<double>(A.Z));
}

template <typename DimT> void launcher(int Init) {
  DimT *A;
  gpuErrCheck(gpuMallocManaged(&A, sizeof(DimT)));
  std::memset(A, 0, sizeof(DimT));

  // Initialize.
  A->X = Init + 3;
  A->Y = Init + 2;
  A->Z = Init + 1;

  testByPtr<<<1, 1>>>(A);
  gpuErrCheck(gpuDeviceSynchronize());
  testByRef<<<1, 1>>>(*A);
  gpuErrCheck(gpuDeviceSynchronize());
  testByVal<<<1, 1>>>(*A);
  gpuErrCheck(gpuDeviceSynchronize());

  gpuErrCheck(gpuFree(A));
}

int main() {
  proteus::init();

  launcher<DimInt>(0);
  launcher<DimFloat>(0);
  launcher<DimDouble>(0);
  launcher<DimMix>(0);
  launcher<DimFill>(0);

  launcher<DimInt>(100);
  launcher<DimFloat>(100);
  launcher<DimDouble>(100);
  launcher<DimFill>(100);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimIntEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimIntEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI8DimFloatEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00@@\00\00\00@\00\00\80?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI8DimFloatEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00@@\00\00\00@\00\00\80?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00@@\00\00\00@\00\00\80?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI9DimDoubleEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI9DimDoubleEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimMixEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimMixEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI6DimMixEvT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI7DimFillEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI7DimFillEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimIntEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"g\00\00\00f\00\00\00e\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimIntEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"g\00\00\00f\00\00\00e\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"g\00\00\00f\00\00\00e\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI8DimFloatEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00\CEB\00\00\CCB\00\00\CAB"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI8DimFloatEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00\CEB\00\00\CCB\00\00\CAB"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value @0 = private {{.*}}constant [12 x i8] c"\00\00\CEB\00\00\CCB\00\00\CAB"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI9DimDoubleEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI9DimDoubleEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value @0 = private {{.*}}constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI7DimFillEvPT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI7DimFillEvRT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private {{.*}}constant [52 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: A X 103.000000 Y 102.000000 Z 101.000000
// CHECK: JitCache hits 0 total 27
// CHECK-COUNT-27: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 27
// CHECK-SECOND: JitStorageCache hits 27 total 27
