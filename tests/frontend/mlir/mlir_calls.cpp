// clang-format off
// RUN: %build/mlir_calls | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");
  auto &Bar = J->addFunction<int(int, float)>("bar");
  auto &Foo = J->addFunction<void()>("foo");

  Bar.beginFunction();
  {
    auto [A, B] = Bar.getArgs();
    (void)B;
    Bar.ret(A);
  }
  Bar.endFunction();

  Foo.beginFunction();
  {
    auto ArgI32 = Foo.defVar<int>(7);
    auto ArgF32 = Foo.defVar<float>(2.5f);
    auto Ret = Foo.call<int(int, float)>("bar", ArgI32, ArgF32);
    auto Sink = Foo.declVar<int>("sink");
    Sink = Ret;
    Foo.ret();
  }
  Foo.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: func.func @bar
// CHECK: func.func @foo
// CHECK: call @bar
// clang-format on
