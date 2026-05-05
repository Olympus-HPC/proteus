#ifndef PROTEUS_RECORDINTERFACE_H
#define PROTEUS_RECORDINTERFACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ProteusRecordedKernel ProteusRecordedKernel;

typedef struct {
  const char *Name;
  const void *HostAddr;
  const void *DevAddr;
  uint64_t Size;
} ProteusRecordedGlobal;

typedef enum {
  PROTEUS_RECORD_OK = 0,
  PROTEUS_RECORD_KERNEL_NOT_FOUND = 1,
  PROTEUS_RECORD_ERROR = 2
} ProteusRecordStatus;

ProteusRecordStatus
__proteus_record_capture_kernel(const void *Kernel,
                                ProteusRecordedKernel **Out);

void __proteus_record_release_kernel(ProteusRecordedKernel *Record);

const char *__proteus_record_kernel_name(const ProteusRecordedKernel *Record);

uint64_t __proteus_record_static_hash(const ProteusRecordedKernel *Record);

const void *__proteus_record_bitcode_data(const ProteusRecordedKernel *Record);

size_t __proteus_record_bitcode_size(const ProteusRecordedKernel *Record);

size_t __proteus_record_global_count(const ProteusRecordedKernel *Record);

ProteusRecordedGlobal
__proteus_record_global_at(const ProteusRecordedKernel *Record, size_t Index);

#ifdef __cplusplus
}
#endif

#endif
