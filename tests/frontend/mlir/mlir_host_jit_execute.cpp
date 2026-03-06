#include <cassert>
#include <cstddef>

#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  JitModule J("host", "mlir");
  auto &Add = J.addFunction<size_t(size_t, size_t)>("add");

  Add.beginFunction();
  {
    auto [A, B] = Add.getArgs();
    Add.ret(A + B);
  }
  Add.endFunction();

  J.compile();

  const size_t Result = Add(40, 2);
  printf("result: %zu\n", Result);
  assert(Result == 42 && "MLIR host JIT execution returned wrong result");

  return 0;
}
