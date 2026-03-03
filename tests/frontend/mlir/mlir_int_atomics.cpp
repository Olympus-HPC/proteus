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

// CHECK: %[[OLDADD:.*]] = memref.atomic_rmw addi %{{.*}}, %{{.*}}[%{{.*}}] :
// (i32, memref<{{.*}}xi32>) -> i32 CHECK: memref.store %[[OLDADD]],
// %[[ADDRES:.*]]{{\[.*\]}} : memref<1xi32>

// CHECK: %[[OLDSUB:.*]] = memref.generic_atomic_rmw %{{.*}}[%{{.*}}] :
// memref<{{.*}}xi32> { CHECK: ^bb0(%[[CURSUB:.*]]: i32): CHECK: %[[NEWSUB:.*]]
// = arith.subi %[[CURSUB]], %{{.*}} : i32 CHECK: memref.atomic_yield
// %[[NEWSUB]] : i32 CHECK: } CHECK: memref.store %[[OLDSUB]],
// %[[SUBRES:.*]]{{\[.*\]}} : memref<1xi32>

// CHECK: %[[OLDMAX:.*]] = memref.atomic_rmw maxs %{{.*}}, %{{.*}}[%{{.*}}] :
// (i32, memref<{{.*}}xi32>) -> i32 CHECK: memref.store %[[OLDMAX]],
// %[[MAXRES:.*]]{{\[.*\]}} : memref<1xi32>

// CHECK: %[[OLDMIN:.*]] = memref.atomic_rmw mins %{{.*}}, %{{.*}}[%{{.*}}] :
// (i32, memref<{{.*}}xi32>) -> i32 CHECK: memref.store %[[OLDMIN]],
// %[[MINRES:.*]]{{\[.*\]}} : memref<1xi32>
