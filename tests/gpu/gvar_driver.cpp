// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/gvar_tracking.%ext | %FILECHECK %s --check-prefixes=CHECK
// clang-format on

void printGVal2(int HValue);
void printGVal1(int HValue);

int main() {
  printGVal1(2);
  printGVal2(3);
}

// NOTE: On my testing system I am getting this output:
// Kernel 1 value is 0
// Kernel 2 value is 3

// CHECK: Kernel 1 value is 2
// CHECK: Kernel 2 value is 3
