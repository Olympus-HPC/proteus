//===-- UtilsCUDA.h -- Utilities header for CUDA --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_UTILS_CUDA_H
#define PROTEUS_UTILS_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvPTXCompiler.h>

#define proteusCudaErrCheck(CALL)                                              \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

#define proteusCuErrCheck(CALL)                                                \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *ErrStr;                                                      \
      cuGetErrorString(err, &ErrStr);                                          \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__, ErrStr);            \
      abort();                                                                 \
    }                                                                          \
  }

#define proteusNvPTXCompilerErrCheck(CALL)                                     \
  {                                                                            \
    nvPTXCompileResult err = CALL;                                             \
    if (err != NVPTXCOMPILE_SUCCESS) {                                         \
      printf("ERROR @ %s:%d ->  %d\n", __FILE__, __LINE__, err);               \
      abort();                                                                 \
    }                                                                          \
  }

#endif
