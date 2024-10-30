//===-- UtilsHIP.h -- Utilities header for HIP --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_UTILS_HIP_H
#define PROTEUS_UTILS_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define hiprtcErrCheck(CALL)                                                   \
  {                                                                            \
    hiprtcResult err = CALL;                                                   \
    if (err != HIPRTC_SUCCESS) {                                               \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hiprtcGetErrorString(err));                                       \
      abort();                                                                 \
    }                                                                          \
  }

#endif
