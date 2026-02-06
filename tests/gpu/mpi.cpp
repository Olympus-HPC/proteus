#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void myDaxpystatic(double A, double *X, double *Y, size_t N) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
  std::size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N) {
    Y[I] += X[I] * A;
  }
}

static inline void launchDaxpy(double A, double *X, double *Y, size_t N) {
  const std::size_t BlockSize = 256;
  const std::size_t GridSize = (N + BlockSize - 1) / BlockSize;
#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL((myDaxpy), dim3(GridSize), dim3(BlockSize), 0, 0, A, X, Y,
                     N);
#elif PROTEUS_ENABLE_CUDA
  void *Args[] = {&A, &X, &Y, &N};
  gpuErrCheck(gpuLaunchKernel((const void *)(myDaxpy), dim3(GridSize),
                              dim3(BlockSize), Args, 0, 0));
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
}

int main(int argc, char **argv) {
  int Provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &Provided);
  if (Provided != MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "MPI_THREAD_MULTIPLE not supported\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int Rank, Size;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Size);

  proteus::init();

  size_t N = 1024;
  double *X;
  double *Y;

  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&Y, sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  if (Rank == 0) {
    std::cout << "Rank " << Rank << " initial Y[10] = " << Y[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int NumIterations = 3;
  for (int Iter = 0; Iter < NumIterations; ++Iter) {
    launchDaxpy(6.2, X, Y, N);
  }
  gpuErrCheck(gpuDeviceSynchronize());

  if (Rank == 0) {
    std::cout << "Rank " << Rank << " final Y[10] = " << Y[10] << std::endl;
  }

  gpuErrCheck(gpuFree(X));
  gpuErrCheck(gpuFree(Y));

  proteus::finalize();

  MPI_Finalize();
  return 0;
}
