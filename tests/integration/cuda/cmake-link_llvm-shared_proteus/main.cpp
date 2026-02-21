#include <proteus/JitInterface.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>

#include <cuda_runtime.h>

#include <stdio.h>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  kernel<<<1, 1>>>();

  return 0;
}
