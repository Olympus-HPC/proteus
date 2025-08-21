// RUN: rm -rf .proteus
// RUN: ./if_else | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule("host");
  auto &Min = J.addFunction<double, double, double>("if.min");
  {
    auto &Arg0 = Min.getArg(0);
    auto &Arg1 = Min.getArg(1);

    Min.beginFunction();
    {
      auto &Ret = Min.declVar<double>("ret");
      Ret = 0;
      Min.IfElse(Arg0 < Arg1, [&]() {
             Ret = Arg0;
             Min.If(Arg0 == 1.0, [&]()
                   {
                     Ret = 500.0;
                    });
             }, /*else*/[&]() {
                 Ret = Arg1;
            });
      Min.ret(Ret);
    }
    Min.endFunction();
  }

  J.compile();
  auto ret = Min(1.0, 2.0);
  std::cout << "R Min " << ret << "\n";

  ret = Min(3.0, 2.0);
  std::cout << "R Min " << ret << "\n";

  proteus::finalize();
  return 0;
}

// CHECK: R Min 500
// CHECK: R Min 2