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

// clang-format off
// CHECK: %[[OLDMAX:.*]] = llvm.atomicrmw fmax %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, f32
// CHECK: memref.store %[[OLDMAX]], %{{.*}}[%{{.*}}] : memref<1xf32>

// CHECK: %[[OLDMIN:.*]] = llvm.atomicrmw fmin %{{.*}}, %{{.*}} seq_cst : !llvm.ptr, f32
// CHECK: memref.store %[[OLDMIN]], %{{.*}}[%{{.*}}] : memref<1xf32>
// clang-format on
