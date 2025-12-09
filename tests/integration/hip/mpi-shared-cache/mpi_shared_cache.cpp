// Test for MPI shared storage cache.
// Verifies that when using PROTEUS_OBJECT_CACHE_CHAIN="mpi-storage",
// only rank 0 writes cache files (no rank prefix in filenames).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <string>

#include <hip/hip_runtime.h>
#include <proteus/JitInterface.hpp>

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

__global__ __attribute__((annotate("jit"))) void testKernel(int RankValue) {
  printf("Kernel from rank %d\n", RankValue);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int Rank, Size;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Size);

  const char *CacheDirEnv = std::getenv("PROTEUS_CACHE_DIR");
  std::string CacheDir = CacheDirEnv ? CacheDirEnv : ".proteus";

  if (Rank == 0) {
    std::filesystem::remove_all(CacheDir);
    std::cout << "Using cache directory: " << CacheDir << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  testKernel<<<1, 1>>>(Rank);
  gpuErrCheck(hipDeviceSynchronize());

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

      std::cout << "Cache files found: " << NumCacheFiles << std::endl;
      std::cout << "Rank-prefixed files: " << NumRankPrefixedFiles << std::endl;

      if (NumCacheFiles >= 1 && NumRankPrefixedFiles == 0) {
        std::cout << "PASS: MPI shared cache test succeeded" << std::endl;
      } else {
        std::cerr << "FAIL: Expected at least 1 cache file with no rank "
                  << "prefix, got " << NumCacheFiles << " files with "
                  << NumRankPrefixedFiles << " rank-prefixed" << std::endl;
        ExitCode = 1;
      }

      std::filesystem::remove_all(CacheDir);
    }
  }

  MPI_Finalize();
  return ExitCode;
}
