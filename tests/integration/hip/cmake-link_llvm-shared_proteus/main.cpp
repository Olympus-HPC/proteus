#include <proteus/JitInterface.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>

#include <hip/hip_runtime.h>

#include <stdio.h>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  kernel<<<1, 1>>>();

  return 0;
}
