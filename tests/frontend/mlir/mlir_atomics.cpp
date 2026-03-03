// clang-format off
// RUN: %build/mlir_atomics | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void(float *)>("mlir_atomics");

  F.beginFunction();
  {
    auto [A] = F.getArgs();
    auto Ptr0 = A + 0;
    auto Two = F.defVar<float>(2.0f);
    auto Three = F.defVar<float>(3.0f);

    auto OldMax = F.atomicMax(Ptr0, Two);
    auto OldMin = F.atomicMin(Ptr0, Three);

    auto OldMaxSlot = F.declVar<float>("old_max_slot");
    auto OldMinSlot = F.declVar<float>("old_min_slot");
    OldMaxSlot = OldMax;
    OldMinSlot = OldMin;

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// CHECK: %[[OLDMAX:.*]] = memref.generic_atomic_rmw
// %[[BUFMAX:.*]][%[[IDXMAX:.*]]] : memref<{{.*}}xf32> { CHECK:
// ^bb0(%[[CURMAX:.*]]: f32): CHECK: %[[NEWMAX:.*]] = arith.maximumf
// %[[CURMAX]], %{{.*}} : f32 CHECK: memref.atomic_yield %[[NEWMAX]] : f32
// CHECK: }
// CHECK: memref.store %[[OLDMAX]], %[[OLDMAXSLOT:.*]]{{\[.*\]}} : memref<1xf32>

// CHECK: %[[OLDMIN:.*]] = memref.generic_atomic_rmw
// %[[BUFMIN:.*]][%[[IDXMIN:.*]]] : memref<{{.*}}xf32> { CHECK:
// ^bb0(%[[CURMIN:.*]]: f32): CHECK: %[[NEWMIN:.*]] = arith.minimumf
// %[[CURMIN]], %{{.*}} : f32 CHECK: memref.atomic_yield %[[NEWMIN]] : f32
// CHECK: }
// CHECK: memref.store %[[OLDMIN]], %[[OLDMINSLOT:.*]]{{\[.*\]}} : memref<1xf32>
