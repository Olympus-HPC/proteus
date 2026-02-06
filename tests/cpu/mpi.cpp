#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <proteus/JitInterface.hpp>
#include <stdio.h>

__attribute__((annotate("jit", 1, 4))) static void
myDaxpy(double A, double *X, double *Y, size_t N) {
  for (std::size_t I{0}; I < N; I++) {
    Y[I] += X[I] * A;
  }
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
  double *X = static_cast<double *>(malloc(sizeof(double) * N));
  double *Y = static_cast<double *>(malloc(sizeof(double) * N));

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
    myDaxpy(6.2, X, Y, N);
  }

  if (Rank == 0) {
    std::cout << "Rank " << Rank << " final Y[10] = " << Y[10] << std::endl;
  }

  free(X);
  free(Y);

  proteus::finalize();

  MPI_Finalize();
  return 0;
}