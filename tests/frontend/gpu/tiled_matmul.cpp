// RUN: rm -rf .proteus
// RUN: ./for | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

#include <cstdlib>
#include <cstring>


using namespace proteus;

#if PROTEUS_ENABLE_HIP

#define TARGET "hip"
#define getThreadIdX builtins::hip::getThreadIdX
#define getThreadIdY builtins::hip::getThreadIdY
#define getThreadIdZ builtins::hip::getThreadIdZ
#define getBlockIdX builtins::hip::getBlockIdX
#define getBlockIdY builtins::hip::getBlockIdY
#define getBlockIdZ builtins::hip::getBlockIdZ
#define getBlockDimX builtins::hip::getBlockDimX
#define getBlockDimY builtins::hip::getBlockDimY
#define getBlockDimZ builtins::hip::getBlockDimZ
#define getGridDimX builtins::hip::getGridDimX
#define getGridDimY builtins::hip::getGridDimY
#define getGridDimZ builtins::hip::getGridDimZ
#define syncThreads builtins::hip::syncThreads

#elif PROTEUS_ENABLE_CUDA

#define TARGET "cuda"
#define getThreadIdX builtins::cuda::getThreadIdX
#define getThreadIdY builtins::cuda::getThreadIdY
#define getThreadIdZ builtins::cuda::getThreadIdZ
#define getBlockIdX builtins::cuda::getBlockIdX
#define getBlockIdY builtins::cuda::getBlockIdY
#define getBlockIdZ builtins::cuda::getBlockIdZ
#define getBlockDimX builtins::cuda::getBlockDimX
#define getBlockDimY builtins::cuda::getBlockDimY
#define getBlockDimZ builtins::cuda::getBlockDimZ
#define getGridDimX builtins::cuda::getGridDimX
#define getGridDimY builtins::cuda::getGridDimY
#define getGridDimZ builtins::cuda::getGridDimZ
#define syncThreads builtins::cuda::syncThreads

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

// clang-format off
// No FileCheck for now; program prints the resulting C matrix

#if PROTEUS_ENABLE_HIP
#ifndef MATMUL_TILE_SIZE
#define MATMUL_TILE_SIZE 32
#endif

__global__ void hipTiledMatmulKernelstatic (const double * __restrict__ A,
                                        const double * __restrict__ B,
                                        double * __restrict__ C,
                                        int N) {
  __shared__ double As[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
  __shared__ double Bs[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];

  int Tx = threadIdx.x;
  int Ty = threadIdx.y;
  int Row = blockIdx.y * MATMUL_TILE_SIZE + Ty;
  int Col = blockIdx.x * MATMUL_TILE_SIZE + Tx;

  double Sum = 0.0;
  int NumTiles = (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE;

  for (int T = 0; T < NumTiles; ++T) {
    int ACol = T * MATMUL_TILE_SIZE + Tx;
    int BRow = T * MATMUL_TILE_SIZE + Ty;

    As[Ty][Tx] = (Row < N && ACol < N) ? A[Row * N + ACol] : 0.0;
    Bs[Ty][Tx] = (BRow < N && Col < N) ? B[BRow * N + Col] : 0.0;
    __syncthreads();

    #pragma unroll
    for (int K = 0; K < MATMUL_TILE_SIZE; ++K) {
      Sum += As[Ty][K] * Bs[K][Tx];
    }
    __syncthreads();
  }

  if (Row < N && Col < N) {
    C[Row * N + Col] = Sum;
  }
}

static inline void hipTiledMatmulLaunch(const double *A,
                                           const double *B,
                                           double *C,
                                           int N) {
  dim3 Block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
  dim3 Grid((N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
            (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
            1);
  hipTiledMatmulKernelstatic<<<Grid, Block>>>(A, B, C, N);
}

// Non-tiled (naive) HIP kernel for matrix multiplication: C = A * B
__global__ void hipNontiledMatmulKernelstatic (const double * __restrict__ A,
                                           const double * __restrict__ B,
                                           double * __restrict__ C,
                                           int N) {
  int Tx = threadIdx.x;
  int Ty = threadIdx.y;
  int Row = blockIdx.y * blockDim.y + Ty;
  int Col = blockIdx.x * blockDim.x + Tx;

  if (Row < N && Col < N) {
    double Sum = 0.0;
    #pragma unroll 1
    for (int K = 0; K < N; ++K) {
      Sum += A[Row * N + K] * B[K * N + Col];
    }
    C[Row * N + Col] = Sum;
  }
}

static inline void hipNontiledMatmulLaunch(const double *A,
                                              const double *B,
                                              double *C,
                                              int N) {
  dim3 Block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
  dim3 Grid((N + Block.x - 1) / Block.x,
            (N + Block.y - 1) / Block.y,
            1);
  hipNontiledMatmulKernelstatic<<<Grid, Block>>>(A, B, C, N);
}
#endif


static auto getMatmulKernel(int N, int TileSize) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<double *, double *, double *>("tiled_matmul");
    auto &F = KernelHandle.F;
  {

    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    F.beginFunction();
    {
        auto Tidx = F.callBuiltin(getThreadIdX);
        auto Bidx = F.callBuiltin(getBlockIdX);
        auto Tidy = F.callBuiltin(getThreadIdY);
        auto Bidy = F.callBuiltin(getBlockIdY);

        auto Row = F.defVar(Bidy * TileSize + Tidy);
        auto Col = F.defVar(Bidx * TileSize + Tidx);

        auto K = F.declVar<int>("K");
        auto Zero = F.defRuntimeConst(0);
        auto One = F.defRuntimeConst(1);
        auto Nvar = F.defRuntimeConst(N);
        auto Accum = F.defVar(0.0);

        F.forLoop({K, Zero, Nvar, One}, [&]() {
            auto AVal = A[Row * N + K];
            auto BVal = B[K * N + Col];
            Accum = Accum + (AVal * BVal);
        });

        auto CIdx = Row * N + Col;
        C[CIdx] = Accum;

        F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}


#if PROTEUS_ENABLE_HIP

// HIP kernel: Register + shared-memory tiled matmul (C = A x B)
// Matches the PJ-DSL kernel configuration via REG_TILE_{M,N}, BLOCK_TILE_{M,N}, K_TILE.
// Ensure tiling defaults are defined before use (duplicated later but guarded by ifndef).
#ifndef REG_TILE_M
#define REG_TILE_M 4
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4
#endif
#ifndef BLOCK_TILE_M
#define BLOCK_TILE_M 64
#endif
#ifndef BLOCK_TILE_N
#define BLOCK_TILE_N 64
#endif
#ifndef K_TILE
#define K_TILE 8
#endif
__global__ void hipRegSharedTiledMatmulKernel(const double * __restrict__ A,
                                              const double * __restrict__ B,
                                              double * __restrict__ C,
                                              int N) {
  // Shared tiles for current K-slice
  __shared__ double AsTile[BLOCK_TILE_M * K_TILE];     // [BLOCK_TILE_M x K_TILE]
  __shared__ double BsTile[K_TILE * BLOCK_TILE_N];     // [K_TILE x BLOCK_TILE_N]

  // Thread indices and block coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Block origin within C
  int blockRow = by * BLOCK_TILE_M;
  int blockCol = bx * BLOCK_TILE_N;

  // Per-thread micro-tile origin within C
  int row0 = blockRow + ty * REG_TILE_M;
  int col0 = blockCol + tx * REG_TILE_N;

  // Register accumulators
  double Creg[REG_TILE_M * REG_TILE_N];
  #pragma unroll
  for (int i = 0; i < REG_TILE_M * REG_TILE_N; ++i) Creg[i] = 0.0;

  // Linear thread id for cooperative loads
  const int threadsX = BLOCK_TILE_N / REG_TILE_N;
  const int tid = ty * threadsX + tx; // 0 .. ((BLOCK_TILE_M/REG_TILE_M)*(BLOCK_TILE_N/REG_TILE_N) - 1)

  // Loop over K dimension in tiles of K_TILE
  for (int kBase = 0; kBase < N; kBase += K_TILE) {
    // Each thread loads two elements of AsTile and BsTile
    int aIdx0 = tid * 2 + 0;
    int aIdx1 = tid * 2 + 1;
    if (aIdx0 < BLOCK_TILE_M * K_TILE) {
      int aRow0 = aIdx0 / K_TILE;
      int aCol0 = aIdx0 % K_TILE;
      int asIdx0 = aRow0 * K_TILE + aCol0;
      int aGlobIdx0 = (blockRow + aRow0) * N + (kBase + aCol0);
      AsTile[asIdx0] = A[aGlobIdx0];
    }
    if (aIdx1 < BLOCK_TILE_M * K_TILE) {
      int aRow1 = aIdx1 / K_TILE;
      int aCol1 = aIdx1 % K_TILE;
      int asIdx1 = aRow1 * K_TILE + aCol1;
      int aGlobIdx1 = (blockRow + aRow1) * N + (kBase + aCol1);
      AsTile[asIdx1] = A[aGlobIdx1];
    }

    int bIdx0 = tid * 2 + 0;
    int bIdx1 = tid * 2 + 1;
    if (bIdx0 < K_TILE * BLOCK_TILE_N) {
      int bRow0 = bIdx0 / BLOCK_TILE_N;
      int bCol0 = bIdx0 % BLOCK_TILE_N;
      int bsIdx0 = bRow0 * BLOCK_TILE_N + bCol0;
      int bGlobIdx0 = (kBase + bRow0) * N + (blockCol + bCol0);
      BsTile[bsIdx0] = B[bGlobIdx0];
    }
    if (bIdx1 < K_TILE * BLOCK_TILE_N) {
      int bRow1 = bIdx1 / BLOCK_TILE_N;
      int bCol1 = bIdx1 % BLOCK_TILE_N;
      int bsIdx1 = bRow1 * BLOCK_TILE_N + bCol1;
      int bGlobIdx1 = (kBase + bRow1) * N + (blockCol + bCol1);
      BsTile[bsIdx1] = B[bGlobIdx1];
    }

    __syncthreads();

    // Compute this micro-tile using the shared tiles and register blocking
    double Areg[REG_TILE_M];
    double Breg[REG_TILE_N];

    #pragma unroll
    for (int kIt = 0; kIt < K_TILE; ++kIt) {
      // Load a REG_TILE_M row from AsTile into registers
      #pragma unroll
      for (int i = 0; i < REG_TILE_M; ++i) {
        int r = ty * REG_TILE_M + i;
        int asIdx = r * K_TILE + kIt;
        Areg[i] = AsTile[asIdx];
      }

      // Load a REG_TILE_N column from BsTile into registers
      #pragma unroll
      for (int j = 0; j < REG_TILE_N; ++j) {
        int c = tx * REG_TILE_N + j;
        int bsIdx = kIt * BLOCK_TILE_N + c;
        Breg[j] = BsTile[bsIdx];
      }

      // FMA on the per-thread micro-tile
      #pragma unroll
      for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
          int cIdx = i * REG_TILE_N + j;
          Creg[cIdx] += Areg[i] * Breg[j];
        }
      }
    }

    __syncthreads();
  }

  // Write back the per-thread micro-tile to C
  #pragma unroll
  for (int i = 0; i < REG_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < REG_TILE_N; ++j) {
      int cGlobIdx = (row0 + i) * N + (col0 + j);
      int rIdx = i * REG_TILE_N + j;
      C[cGlobIdx] = Creg[rIdx];
    }
  }
}

static inline void hipRegSharedTiledMatmulLaunch(const double *A,
                                                 const double *B,
                                                 double *C,
                                                 int N) {
  dim3 Block(BLOCK_TILE_N / REG_TILE_N, BLOCK_TILE_M / REG_TILE_M, 1);
  dim3 Grid(N / BLOCK_TILE_N, N / BLOCK_TILE_M, 1);
  hipRegSharedTiledMatmulKernel<<<Grid, Block>>>(A, B, C, N);
}

#endif // PROTEUS_ENABLE_HIP

// Register + shared-memory tiled JIT kernel (C = A x B) for square N x N, double.
// Configuration mirrors the HIP kernel: 64x64 block tile, 16x16 threads,
// 4x4 per-thread micro-tile, K tile = 8 (defaults; overridable via macros).
#ifndef REG_TILE_M
#define REG_TILE_M 4
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4
#endif
#ifndef BLOCK_TILE_M
#define BLOCK_TILE_M 64
#endif
#ifndef BLOCK_TILE_N
#define BLOCK_TILE_N 64
#endif
#ifndef K_TILE
#define K_TILE 8
#endif

static auto getRegSharedTiledMatmulKernel(int N) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<double *, double *, double*>("reg_shared_tiled_matmul");
  auto &F = KernelHandle.F;
  {
    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);

    // Shared tiles for current K-slice
    auto &AsTile = F.declArray<double>(BLOCK_TILE_M * K_TILE, AddressSpace::SHARED);
    auto &BsTile = F.declArray<double>(K_TILE * BLOCK_TILE_N, AddressSpace::SHARED);

    auto &Areg = F.declArray<double>(REG_TILE_M);
    auto &Breg = F.declArray<double>(REG_TILE_N);
    auto &Creg = F.declArray<double>(REG_TILE_M * REG_TILE_N);

    F.beginFunction();
    {
      auto Tidx = F.callBuiltin(getThreadIdX);
      auto Tidy = F.callBuiltin(getThreadIdY);
      auto Bidx = F.callBuiltin(getBlockIdX);
      auto Bidy = F.callBuiltin(getBlockIdY);

      // Constants
      auto Nvar = F.defRuntimeConst(N);
      auto Two = F.defRuntimeConst(2);
      auto Zero = F.defRuntimeConst(0);
      auto One = F.defRuntimeConst(1);
      auto RegTileM = F.defRuntimeConst(REG_TILE_M);
      auto RegTileN = F.defRuntimeConst(REG_TILE_N);
      auto BlockTileM = F.defRuntimeConst(BLOCK_TILE_M);
      auto BlockTileN = F.defRuntimeConst(BLOCK_TILE_N);
      auto KTile = F.defRuntimeConst(K_TILE);
      auto ThreadsX = F.defRuntimeConst(BLOCK_TILE_N / REG_TILE_N);

      // Block origin in C
      auto BlockRow = F.defVar(Bidy * BlockTileM);
      auto BlockCol = F.defVar(Bidx * BlockTileN);

      // Per-thread micro tile origin in C
      auto Row0 = F.defVar(BlockRow + Tidy * RegTileM);
      auto Col0 = F.defVar(BlockCol + Tidx * RegTileN);

      // Zero accumulators
      {
        auto I = F.declVar<int>("i");
        auto J = F.declVar<int>("j");
        F.forLoop({I, Zero, RegTileM, One}, [&]() {
          F.forLoop({J, Zero, RegTileN, One}, [&]() {
            auto Cidx = I * RegTileN + J;
            Creg[Cidx] = 0.0;
          });
        });
      }

      // Loop over K dimension in tiles of K_TILE
      auto KBase = F.declVar<int>("KBase");
      F.forLoop({KBase, Zero, Nvar, KTile}, [&]() {
        // Cooperative load of A and B tiles into shared memory.
        auto Tid = F.defVar(Tidy * ThreadsX + Tidx); // 0 .. (BLOCK_TILE_M/REG_TILE_M*BLOCK_TILE_N/REG_TILE_N - 1)

        // Load A tile: size [BLOCK_TILE_M x K_TILE]
        auto AIdx1 = Tid * Two + One;
        auto AIdx0 = Tid * Two + Zero;
        auto ARow0 = AIdx0 / KTile;
        auto ACol0 = AIdx0 % KTile;
        auto ARow1 = AIdx1 / KTile;
        auto ACol1 = AIdx1 % KTile;
        auto AsIdx0 = ARow0 * KTile + ACol0;
        auto AsIdx1 = ARow1 * KTile + ACol1;
        auto AGlobIdx0 = (BlockRow + ARow0) * N + (KBase + ACol0);
        auto AGlobIdx1 = (BlockRow + ARow1) * N + (KBase + ACol1);
        AsTile[AsIdx0] = A[AGlobIdx0];
        AsTile[AsIdx1] = A[AGlobIdx1];

        // Load B tile: size [K_TILE x BLOCK_TILE_N]
        auto BIdx0 = Tid * Two + Zero;
        auto BIdx1 = Tid * Two + One;
        auto BRow0 = BIdx0 / BlockTileN;
        auto BCol0 = BIdx0 % BlockTileN;
        auto BRow1 = BIdx1 / BlockTileN;
        auto BCol1 = BIdx1 % BlockTileN;
        auto BsIdx0 = BRow0 * BlockTileN + BCol0;
        auto BsIdx1 = BRow1 * BlockTileN + BCol1;
        auto BGlobIdx0 = (KBase + BRow0) * N + (BlockCol + BCol0);
        auto BGlobIdx1 = (KBase + BRow1) * N + (BlockCol + BCol1);
        BsTile[BsIdx0] = B[BGlobIdx0];
        BsTile[BsIdx1] = B[BGlobIdx1];

        F.callBuiltin(syncThreads);

        // Compute this micro-tile using the shared tiles and register blocking
        auto KIt = F.declVar<int>("KIt");
        F.forLoop({KIt, Zero, KTile, One}, [&]() {
          // Load rows/cols into registers
          auto I = F.declVar<int>("i");
          auto J = F.declVar<int>("j");

          F.forLoop({I, Zero, RegTileM, One}, [&]() {
            auto r = Tidy * RegTileM + I;
            auto asIdx = r * KTile + KIt;
            Areg[I] = AsTile[asIdx];
          });

          F.forLoop({J, Zero, RegTileN, One}, [&]() {
            auto c = Tidx * RegTileN + J;
            auto bsIdx = KIt * BlockTileN + c;
            Breg[J] = BsTile[bsIdx];
          });

          // FMA on the micro-tile
          auto Ii = F.declVar<int>("ii");
          auto Jj = F.declVar<int>("jj");
          F.forLoop({Ii, Zero, RegTileM, One}, [&]() {
            F.forLoop({Jj, Zero, RegTileN, One}, [&]() {
              auto Cidx = Ii * RegTileN + Jj;
              Creg[Cidx] = Creg[Cidx] + (Areg[Ii] * Breg[Jj]);
            });
          });
        });

        F.callBuiltin(syncThreads);
      });

      // Write back the per-thread micro-tile to C
      {
        auto I = F.declVar<int>("i");
        auto J = F.declVar<int>("j");
        F.forLoop({I, Zero, RegTileM, One}, [&]() {
          F.forLoop({J, Zero, RegTileN, One}, [&]() {
            auto Cidx = (Row0 + I) * N + (Col0 + J);
            auto Ridx = I * RegTileN + J;
            C[Cidx] = Creg[Ridx];
          });
        });
      }

      F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

int main(int argc, char** argv) {
  proteus::init();
  unsigned int N = 8192;
  int NumTrials = 5;
  bool DoVerify = true;
  std::string KernelType = "jit_regtiled";

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--N") || !std::strcmp(argv[i], "-n")) {
      if (i + 1 < argc) {
        N = static_cast<unsigned int>(std::atoi(argv[++i]));
      }
    } else if (!std::strcmp(argv[i], "--trials") || !std::strcmp(argv[i], "-t")) {
      if (i + 1 < argc) {
        NumTrials = std::atoi(argv[++i]);
      }
    } else if (!std::strcmp(argv[i], "--kernel")) {
      if (i + 1 < argc) {
        KernelType = argv[++i];
        if (KernelType != "hip" && KernelType != "hip_regtiled" &&
            KernelType != "jit" && KernelType != "jit_regtiled") {
          std::cerr << "Error: Invalid kernel type '" << KernelType
                    << "'. Valid options: hip, hip_regtiled, jit, jit_regtiled\n";
          return 1;
        }
      }
    } else if (!std::strcmp(argv[i], "--verify")) {
      DoVerify = true;
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      DoVerify = false;
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [-n|--N N] [-t|--trials T] [--kernel KERNEL] [--verify|--no-verify]\n"
                << "  KERNEL: hip, hip_regtiled, jit, jit_regtiled (default: jit_regtiled)\n";
      return 0;
    }
  }
  std::cout << "Configuration: (N, NumTrials, DoVerify, Kernel) = (" << N << ", " << NumTrials << ", " << DoVerify << ", " << KernelType << ")" << std::endl;

  // Host allocations
  double *AH = (double *)new double[N * N];
  double *BH = (double *)new double[N * N];
  double *CH = (double *)new double[N * N];

  // Device allocations
  double *AD;
  double *BD;
  double *CD;
  size_t Bytes = sizeof(double) * N * N;
  gpuErrCheck(gpuMalloc((void **)&AD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&BD, Bytes));
  gpuErrCheck(gpuMalloc((void **)&CD, Bytes));

    for (int I = 0; I < N; I++) {
      for (int J = 0; J < N; J++) {
        AH[I*N+J] = (1.0);
        BH[I*N+J] = (1.0);
        CH[I*N+J] = 0.0;
      }
    }

  // Stage inputs to device
  gpuErrCheck(gpuMemcpy(AD, AH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(BD, BH, Bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(CD, CH, Bytes, gpuMemcpyHostToDevice));

  // Kernel execution based on type
  if (KernelType == "jit_regtiled") {
    auto [JitMod, KernelHandle] = getRegSharedTiledMatmulKernel(N);
    JitMod->compile();
    gpuErrCheck(KernelHandle.launch({N / BLOCK_TILE_N, N / BLOCK_TILE_M, 1}, {BLOCK_TILE_N / REG_TILE_N, BLOCK_TILE_M / REG_TILE_M, 1}, 0, nullptr, CD, AD, BD));
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      gpuErrCheck(KernelHandle.launch({N / BLOCK_TILE_N, N / BLOCK_TILE_M, 1}, {BLOCK_TILE_N / REG_TILE_N, BLOCK_TILE_M / REG_TILE_M, 1}, 0, nullptr, CD, AD, BD));
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else if (KernelType == "jit") {
    auto [JitMod, KernelHandle] = getMatmulKernel(N, MATMUL_TILE_SIZE);
    JitMod->compile();
    gpuErrCheck(KernelHandle.launch({(N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, 1}, {MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1}, 0, nullptr, CD, AD, BD));
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      gpuErrCheck(KernelHandle.launch({(N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, 1}, {MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1}, 0, nullptr, CD, AD, BD));
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else if (KernelType == "hip_regtiled") {
    hipRegSharedTiledMatmulLaunch(AD, BD, CD, N);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      hipRegSharedTiledMatmulLaunch(AD, BD, CD, N);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';

  } else if (KernelType == "hip") {
    hipNontiledMatmulLaunch(AD, BD, CD, N);
    gpuErrCheck(gpuDeviceSynchronize());

    // Timed trials
    double TotalMs = 0.0;
    auto Start = std::chrono::high_resolution_clock::now();
    for (int T = 0; T < NumTrials; ++T) {
      hipNontiledMatmulLaunch(AD, BD, CD, N);
    }
    gpuErrCheck(gpuDeviceSynchronize());
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Ms = End - Start;
    TotalMs = Ms.count();
    double AvgMs = TotalMs / static_cast<double>(NumTrials);
    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms" << '\n';
  }

  // Final result back to host after timing loop
  if (DoVerify) {
    gpuErrCheck(gpuMemcpy(CH, CD, Bytes, gpuMemcpyDeviceToHost));
    for (int I = 0; I < N; I++) {
      for (int J = 0; J < N; J++) {
        if (CH[I*N + J] != N) {
          std::cout << "ERROR: " << "C_h[" << I << "][" << J << "] = " << CH[I*N + J] << ", expected " << N << '\n';
          break;
        }
      }
    }
  }

  // Cleanup device and host memory
  gpuErrCheck(gpuFree(AD));
  gpuErrCheck(gpuFree(BD));
  gpuErrCheck(gpuFree(CD));
  delete[] AH;
  delete[] BH;
  delete[] CH;

  proteus::finalize();
  return 0;

}
