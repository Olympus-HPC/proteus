// Stress test for MPI shared storage cache.
// Verifies that when using PROTEUS_OBJECT_CACHE_CHAIN="mpi-storage",
// only rank 0 writes cache files (no rank prefix in filenames).
// Creates many specializations: 5 from testKernel + 6*NumRanks from
// configKernel.

#include <proteus/JitInterface.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mpi.h>
#include <string>

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

__global__ void testKernel(int Mode) {
  proteus::jit_arg(Mode);
  printf("Kernel mode=%d\n", Mode);
}

__global__ void configKernel(int A, int B, int C) {
  proteus::jit_arg(A);
  proteus::jit_arg(B);
  proteus::jit_arg(C);
  printf("Config: A=%d B=%d C=%d sum=%d\n", A, B, C, A + B + C);
}

int main(int argc, char **argv) {
  int Provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &Provided);
  if (Provided != MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "MPI_THREAD_MULTIPLE not supported\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Init proteus AFTER MPI.
  int Rank, Size;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Size);

  const char *CacheDirEnv = std::getenv("PROTEUS_CACHE_DIR");
  std::string CacheDir = CacheDirEnv ? CacheDirEnv : ".proteus";

  if (Rank == 0) {
    std::cout << "Using cache directory: " << CacheDir << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for (int Mode = 0; Mode < 5; ++Mode) {
    testKernel<<<1, 1>>>(Mode);
    gpuErrCheck(hipDeviceSynchronize());
  }

  for (int A = 0; A < 3; ++A) {
    for (int B = 0; B < 2; ++B) {
      configKernel<<<1, 1>>>(A, B, Rank);
      gpuErrCheck(hipDeviceSynchronize());
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int ExitCode = 0;

  if (Rank == 0) {
    int NumCacheFiles = 0;
    int NumRankPrefixedFiles = 0;

    if (!std::filesystem::exists(CacheDir)) {
      std::cerr << "FAIL: Cache directory does not exist: " << CacheDir
                << std::endl;
      ExitCode = 1;
    } else {
      for (const auto &Entry : std::filesystem::directory_iterator(CacheDir)) {
        std::string Filename = Entry.path().filename().string();

        if (Filename.find("cache-jit-") != std::string::npos) {
          NumCacheFiles++;
          std::cout << "Found cache file: " << Filename << std::endl;

          if (Filename.size() > 0 && std::isdigit(Filename[0]) &&
              Filename.find("-cache-jit-") == 1) {
            NumRankPrefixedFiles++;
            std::cerr << "ERROR: Found rank-prefixed cache file: " << Filename
                      << std::endl;
          }
        }
      }

      int ExpectedMax = 5 + 6 * Size;
      std::cout << "Cache files found: " << NumCacheFiles << std::endl;
      std::cout << "Expected (max possible): " << ExpectedMax << std::endl;
      std::cout << "Rank-prefixed files: " << NumRankPrefixedFiles << std::endl;

      if (NumRankPrefixedFiles > 0) {
        std::cerr << "FAIL: Found " << NumRankPrefixedFiles
                  << " rank-prefixed cache files (expected 0)" << std::endl;
        ExitCode = 1;
      } else if (NumCacheFiles == 0) {
        std::cerr << "FAIL: No cache files found" << std::endl;
        ExitCode = 1;
      } else {
        std::cout << "PASS: MPI shared cache test succeeded (" << NumCacheFiles
                  << "/" << ExpectedMax << " specializations cached)"
                  << std::endl;
      }
    }
  }

  MPI_Finalize();
  return ExitCode;
}
