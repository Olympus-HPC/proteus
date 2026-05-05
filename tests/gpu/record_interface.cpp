// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/record_interface.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <proteus/JitInterface.h>
#include <proteus/RecordInterface.h>

__device__ int RecordInterfaceGlobal = 17;

__global__ __attribute__((annotate("jit"))) void record_interface_kernel() {
  RecordInterfaceGlobal += 1;
  printf("Record interface kernel %d\n", RecordInterfaceGlobal);
}

__global__ void plain_kernel() { printf("Plain kernel\n"); }

static void require(bool Condition, const char *Message) {
  if (!Condition) {
    std::printf("FAIL: %s\n", Message);
    std::abort();
  }
}

int main() {
  ProteusRecordedKernel *Record = nullptr;
  auto Status = __proteus_record_capture_kernel(
      reinterpret_cast<const void *>(record_interface_kernel), &Record);
  require(Status == PROTEUS_RECORD_OK, "expected capture to succeed");
  require(Record != nullptr, "expected non-null record");

  const char *Name = __proteus_record_kernel_name(Record);
  uint64_t Hash = __proteus_record_static_hash(Record);
  const void *BitcodeData = __proteus_record_bitcode_data(Record);
  size_t BitcodeSize = __proteus_record_bitcode_size(Record);
  size_t GlobalCount = __proteus_record_global_count(Record);

  require(Name && std::strlen(Name) > 0, "expected kernel name");
  require(Hash != 0, "expected static hash");
  require(BitcodeData != nullptr, "expected bitcode data");
  require(BitcodeSize > 0, "expected bitcode bytes");
  require(GlobalCount > 0, "expected globals");

  bool FoundGlobal = false;
  for (size_t I = 0; I < GlobalCount; ++I) {
    ProteusRecordedGlobal Global = __proteus_record_global_at(Record, I);
    require(Global.Name != nullptr, "expected global name");
    require(Global.DevAddr != nullptr, "expected global device address");
    require(Global.Size > 0, "expected global size");
    FoundGlobal |= std::strcmp(Global.Name, "RecordInterfaceGlobal") == 0;
  }
  require(FoundGlobal, "expected RecordInterfaceGlobal");

  std::printf("Captured %s hash %llu bitcode %zu globals %zu\n", Name,
              static_cast<unsigned long long>(Hash), BitcodeSize, GlobalCount);
  __proteus_record_release_kernel(Record);

  Record = nullptr;
  Status = __proteus_record_capture_kernel(
      reinterpret_cast<const void *>(plain_kernel), &Record);
  require(Status == PROTEUS_RECORD_KERNEL_NOT_FOUND,
          "expected plain kernel to be ignored");
  require(Record == nullptr, "expected no record for plain kernel");

  record_interface_kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  plain_kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK: Captured {{.*}}record_interface_kernel{{.*}} hash {{[0-9]+}} bitcode {{[1-9][0-9]*}} globals {{[1-9][0-9]*}}
// CHECK: Record interface kernel 18
// CHECK: Plain kernel
