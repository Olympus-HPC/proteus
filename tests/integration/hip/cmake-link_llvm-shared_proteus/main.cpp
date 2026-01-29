#include <proteus/JitInterface.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>

#include <hip/hip_runtime.h>

#include <stdio.h>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  proteus::init();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  kernel<<<1, 1>>>();

  proteus::finalize();
  return 0;
}
