// clang-format off
// RUN: %build/mlir_int_atomics | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void(int *)>("mlir_int_atomics");

  F.beginFunction();
  {
    auto [A] = F.getArgs();
    auto Ptr0 = A + 0;
    auto One = F.defVar<int>(1);
    auto Two = F.defVar<int>(2);

    auto OldAdd = F.atomicAdd(Ptr0, One);
    auto OldSub = F.atomicSub(Ptr0, One);
    auto OldMax = F.atomicMax(Ptr0, Two);
    auto OldMin = F.atomicMin(Ptr0, Two);

    auto KeepAdd = F.declVar<int>("keep_add");
    auto KeepSub = F.declVar<int>("keep_sub");
    auto KeepMax = F.declVar<int>("keep_max");
    auto KeepMin = F.declVar<int>("keep_min");
    KeepAdd = OldAdd;
    KeepSub = OldSub;
    KeepMax = OldMax;
    KeepMin = OldMin;

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: %[[OLDADD:.*]] = llvm.atomicrmw add %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, i32
// CHECK: memref.store %[[OLDADD]], %{{.*}}[%{{.*}}] : memref<1xi32>

// CHECK: %[[OLDSUB:.*]] = llvm.atomicrmw sub %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, i32
// CHECK: memref.store %[[OLDSUB]], %{{.*}}[%{{.*}}] : memref<1xi32>

// CHECK: %[[OLDMAX:.*]] = llvm.atomicrmw max %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, i32
// CHECK: memref.store %[[OLDMAX]], %{{.*}}[%{{.*}}] : memref<1xi32>

// CHECK: %[[OLDMIN:.*]] = llvm.atomicrmw min %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, i32
// CHECK: memref.store %[[OLDMIN]], %{{.*}}[%{{.*}}] : memref<1xi32>
// clang-format on
