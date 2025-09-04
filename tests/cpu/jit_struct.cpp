// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./jit_struct | %FILECHECK %s --check-prefixes=CHECK,CHECK-%target_arch,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./jit_struct | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>

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
  int Fill[16];
};

template <typename DimT> void testByPtr(DimT *A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A->X),
         static_cast<double>(A->Y), static_cast<double>(A->Z));
}
template <typename DimT> void testByRef(DimT &A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A.X),
         static_cast<double>(A.Y), static_cast<double>(A.Z));
}

template <typename DimT> void testByVal(DimT A) {
  proteus::jit_object(A);
  printf("A X %lf Y %lf Z %lf\n", static_cast<double>(A.X),
         static_cast<double>(A.Y), static_cast<double>(A.Z));
}

template <typename DimT> void launcher(int Init) {
  DimT A;

  std::memset(&A, 0, sizeof(DimT));

  // Initialize.
  A.X = Init + 3;
  A.Y = Init + 2;
  A.Z = Init + 1;

  testByPtr(&A);
  testByRef(A);
  testByVal(A);
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
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimIntEvPT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimIntEvRT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value i64 8589934595
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 1 with value i32 1
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value [2 x i64] [i64 8589934595, i64 {{[0-9]+}}]
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI8DimFloatEvPT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\00\00@@\00\00\00@\00\00\80?"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI8DimFloatEvRT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\00\00@@\00\00\00@\00\00\80?"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value <2 x float> <float 3.000000e+00, float 2.000000e+00>
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 1 with value float 1.000000e+00
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value [3 x float] [float 3.000000e+00, float 2.000000e+00, float 1.000000e+00]
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI9DimDoubleEvPT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI9DimDoubleEvRT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\00\08@\00\00\00\00\00\00\00@\00\00\00\00\00\00\F0?"
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value [3 x double] [double 3.000000e+00, double 2.000000e+00, double 1.000000e+00]
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimMixEvPT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimMixEvRT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI6DimMixEvT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\80?\00\00\00\00"
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI6DimMixEvT_ ArgNo 0 with value [3 x i64] [i64 3, i64 4611686018427387904, i64 {{[0-9]+}}]
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI7DimFillEvPT_ ArgNo 0 with value @0 = private constant [76 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI7DimFillEvRT_ ArgNo 0 with value @0 = private constant [76 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private constant [76 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private constant [76 x i8] c"\03\00\00\00\02\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 3.000000 Y 2.000000 Z 1.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI6DimIntEvPT_ ArgNo 0 with value @0 = private constant [12 x i8] c"g\00\00\00f\00\00\00e\00\00\00"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI6DimIntEvRT_ ArgNo 0 with value @0 = private constant [12 x i8] c"g\00\00\00f\00\00\00e\00\00\00"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value i64 438086664295
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 1 with value i32 101
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI6DimIntEvT_ ArgNo 0 with value [2 x i64] [i64 438086664295, i64 {{[0-9]+}}]
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI8DimFloatEvPT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\00\00\CEB\00\00\CCB\00\00\CAB"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI8DimFloatEvRT_ ArgNo 0 with value @0 = private constant [12 x i8] c"\00\00\CEB\00\00\CCB\00\00\CAB"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value <2 x float> <float 1.030000e+02, float 1.020000e+02>
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 1 with value float 1.010000e+02
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI8DimFloatEvT_ ArgNo 0 with value [3 x float] [float 1.030000e+02, float 1.020000e+02, float 1.010000e+02]
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI9DimDoubleEvPT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI9DimDoubleEvRT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value @0 = private constant [24 x i8] c"\00\00\00\00\00\C0Y@\00\00\00\00\00\80Y@\00\00\00\00\00@Y@"
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI9DimDoubleEvT_ ArgNo 0 with value [3 x double] [double 1.030000e+02, double 1.020000e+02, double 1.010000e+02]
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByPtrI7DimFillEvPT_ ArgNo 0 with value @0 = private constant [76 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9testByRefI7DimFillEvRT_ ArgNo 0 with value @0 = private constant [76 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK-x86_64: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private constant [76 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK-ppc64le: [ArgSpec] Replaced Function _Z9testByValI7DimFillEvT_ ArgNo 0 with value @0 = private constant [76 x i8] c"g\00\00\00f\00\00\00e\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
// CHECK: X 103.000000 Y 102.000000 Z 101.000000
// CHECK: JitCache hits 0 total 27
// CHECK-COUNT-27: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 27
// CHECK-SECOND: JitStorageCache hits 27 total 27
