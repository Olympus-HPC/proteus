// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/vars | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/vars | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <cstdio>

#include <proteus/JitFrontend.h>
#include <proteus/JitInterface.h>

int main() {
  // Test declVars with anonymous variables.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("declVarsAnonymous");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.declVars<int, double, float>();
      I = 10;
      D = 20.5;
      Fl = 30.25f;
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    J.compile();

    double Result[3] = {0.0, 0.0, 0.0};
    F(Result);

    std::cout << "declVarsAnon[0] = " << Result[0] << "\n";
    std::cout << "declVarsAnon[1] = " << Result[1] << "\n";
    std::cout << "declVarsAnon[2] = " << Result[2] << "\n";
  }

  // Test declVars with named variables.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("declVarsNamed");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.declVars<int, double, float>("I", "D", "F");
      I = 100;
      D = 200.5;
      Fl = 300.25f;
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double Result[3] = {0.0, 0.0, 0.0};
    F(Result);

    std::cout << "declVarsNamed[0] = " << Result[0] << "\n";
    std::cout << "declVarsNamed[1] = " << Result[1] << "\n";
    std::cout << "declVarsNamed[2] = " << Result[2] << "\n";
  }

  // Test defVars with raw values.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("defVarsRaw");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.defVars(10, 20.5, 30.25f);
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    J.compile();

    double Result[3] = {0.0, 0.0, 0.0};
    F(Result);

    std::cout << "defVarsRaw[0] = " << Result[0] << "\n";
    std::cout << "defVarsRaw[1] = " << Result[1] << "\n";
    std::cout << "defVarsRaw[2] = " << Result[2] << "\n";
  }

  // Test defVars with named pairs.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("defVarsNamed");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D, Fl] = F.defVars(std::pair{50, "I"}, std::pair{60.5, "D"},
                                  std::pair{70.25f, "F"});
      Arg[0] = I;
      Arg[1] = D;
      Arg[2] = Fl;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double Result[3] = {0.0, 0.0, 0.0};
    F(Result);

    std::cout << "defVarsNamed[0] = " << Result[0] << "\n";
    std::cout << "defVarsNamed[1] = " << Result[1] << "\n";
    std::cout << "defVarsNamed[2] = " << Result[2] << "\n";
  }

  // Test defVars with mixed raw and named.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("defVarsMixed");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto [I, D] = F.defVars(123, std::pair{456.789, "D"});
      Arg[0] = I;
      Arg[1] = D;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double Result[2] = {0.0, 0.0};
    F(Result);

    std::cout << "defVarsMixed[0] = " << Result[0] << "\n";
    std::cout << "defVarsMixed[1] = " << Result[1] << "\n";
  }

  // Test defVar pair overload with Var<T> returns Var<T>, not Var<Var<T>>.
  {
    auto J = proteus::JitModule();
    auto &F = J.addFunction<void(double *)>("defVarPairVar");

    auto &Arg = F.getArg<0>();
    F.beginFunction();
    {
      auto V1 = F.defVar(3.14159, "original");
      auto V2 = F.defVar(std::pair{V1, "copy"});
      Arg[0] = V2;
      F.ret();
    }
    F.endFunction();

    std::cout.flush();
    J.print();
    fflush(stdout);

    J.compile();

    double Result[1] = {0.0};
    F(Result);

    std::cout << "defVarPairVar[0] = " << Result[0] << "\n";
  }

  return 0;
}

// clang-format off

// CHECK: declVarsAnon[0] = 10
// CHECK-NEXT: declVarsAnon[1] = 20.5
// CHECK-NEXT: declVarsAnon[2] = 30.25

// CHECK: define {{.*}} @declVarsNamed
// CHECK-DAG: %I = alloca i32
// CHECK-DAG: %D = alloca double
// CHECK-DAG: %F = alloca float
// CHECK: declVarsNamed[0] = 100
// CHECK-NEXT: declVarsNamed[1] = 200.5
// CHECK-NEXT: declVarsNamed[2] = 300.25

// CHECK: defVarsRaw[0] = 10
// CHECK-NEXT: defVarsRaw[1] = 20.5
// CHECK-NEXT: defVarsRaw[2] = 30.25

// CHECK: define {{.*}} @defVarsNamed
// CHECK-DAG: %I = alloca i32
// CHECK-DAG: %D = alloca double
// CHECK-DAG: %F = alloca float
// CHECK: defVarsNamed[0] = 50
// CHECK-NEXT: defVarsNamed[1] = 60.5
// CHECK-NEXT: defVarsNamed[2] = 70.25

// CHECK: define {{.*}} @defVarsMixed
// CHECK-DAG: %D = alloca double
// CHECK: defVarsMixed[0] = 123
// CHECK-NEXT: defVarsMixed[1] = 456.789

// CHECK: define {{.*}} @defVarPairVar
// CHECK-DAG: %copy = alloca double
// CHECK: defVarPairVar[0] = 3.14159

// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 6
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 6 accesses 6
