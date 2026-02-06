#include <cstdio>

#include "proteus/CppJitModule.h"
#include <proteus/JitInterface.h>

using namespace proteus;

int main() {
  proteus::init();
  const char *Code = R"cpp(
        #include <cstdio>
        extern "C" void foo() {
            printf("hello world!\n");
})cpp";

  CppJitModule CJM{"host", Code, {"-fplugin=./libClangPlugin.so"}};
  auto Foo = CJM.getFunction<void()>("foo");
  Foo.run();

  proteus::finalize();
  return 0;
}
