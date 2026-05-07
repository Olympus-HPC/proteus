// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_metadata.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <cstdlib>
#include <proteus/JitInterface.h>
#include <proteus/KernelMetadata.h>

__device__ int KernelMetadataGlobal = 17;

__global__ __attribute__((annotate("jit"))) void kernel_metadata_kernel() {
  KernelMetadataGlobal += 1;
  printf("Kernel metadata kernel %d\n", KernelMetadataGlobal);
}

__global__ void plain_kernel() { printf("Plain kernel\n"); }

static void require(bool Condition, const char *Message) {
  if (!Condition) {
    std::printf("FAIL: %s\n", Message);
    std::abort();
  }
}

int main() {
  auto Metadata = proteus::runtime::captureKernelMetadata(
      reinterpret_cast<const void *>(kernel_metadata_kernel));
  require(Metadata.has_value(), "expected capture to succeed");

  const auto &Record = *Metadata;
  const auto &Name = Record.getName();
  uint64_t Hash = Record.getStaticHash();
  const auto &Bitcode = Record.getBitcode();
  const auto &Globals = Record.getGlobals();

  require(!Name.empty(), "expected kernel name");
  require(Hash != 0, "expected static hash");
  require(!Bitcode.empty(), "expected bitcode bytes");
  require(!Globals.empty(), "expected globals");

  auto Global = Globals.find("KernelMetadataGlobal");
  require(Global != Globals.end(), "expected KernelMetadataGlobal");
  require(Global->second.DevAddr != nullptr, "expected global device address");
  require(Global->second.VarSize > 0, "expected global size");

  std::printf("Captured %s hash %llu bitcode %zu globals %zu\n", Name.c_str(),
              static_cast<unsigned long long>(Hash), Bitcode.size(),
              Globals.size());

  auto PlainMetadata = proteus::runtime::captureKernelMetadata(
      reinterpret_cast<const void *>(plain_kernel));
  require(!PlainMetadata, "expected plain kernel to be ignored");

  kernel_metadata_kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  plain_kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK: Captured {{.*}}kernel_metadata_kernel{{.*}} hash {{[0-9]+}} bitcode {{[1-9][0-9]*}} globals {{[1-9][0-9]*}}
// CHECK: Kernel metadata kernel 18
// CHECK: Plain kernel
