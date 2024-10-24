#ifndef PROTEUS_UTILS_CUDA_H
#define PROTEUS_UTILS_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvPTXCompiler.h>

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *ErrStr;                                                      \
      cuGetErrorString(err, &ErrStr);                                          \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__, ErrStr);            \
      abort();                                                                 \
    }                                                                          \
  }

#define nvPTXCompilerErrCheck(CALL)                                            \
  {                                                                            \
    nvPTXCompileResult err = CALL;                                             \
    if (err != NVPTXCOMPILE_SUCCESS) {                                         \
      printf("ERROR @ %s:%d ->  %d\n", __FILE__, __LINE__, err);               \
      abort();                                                                 \
    }                                                                          \
  }

#endif
