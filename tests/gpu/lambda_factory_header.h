#include <cstdio>
#include <proteus/JitInterface.h>

inline auto declareLambda(int rc1, int rc2) {
  return [=, C1 = proteus::jit_variable(rc1), C2 = proteus::jit_variable(rc2)]()
             __attribute__((annotate("jit"))) {
               printf("Integer = %d\n", C1);
               printf("Integer = %d\n", C2);
             };
}
