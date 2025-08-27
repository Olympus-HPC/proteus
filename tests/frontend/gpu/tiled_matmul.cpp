// RUN: rm -rf .proteus
// RUN: ./for | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

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


static auto getTiledMatmulKernel(int N, int TileSize) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<double *, double *, double *>("tiled_matmul");
    auto &F = KernelHandle.F;
  {

    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);
    auto &AsTile = F.declArray<double>(TileSize * TileSize, Array::AddressSpace::SHARED);
    auto &BsTile = F.declArray<double>(TileSize * TileSize, Array::AddressSpace::SHARED);

    F.beginFunction();
    {

        auto Tidx = F.callBuiltin(getThreadIdX);
        auto Bidx = F.callBuiltin(getBlockIdX);
        auto Tidy = F.callBuiltin(getThreadIdY);
        auto Bidy = F.callBuiltin(getBlockIdY);

        auto Row = F.declVar<int>("Row");
        auto Col = F.declVar<int>("Col");

        Row = Bidy * TileSize + Tidy;
        Col = Bidx * TileSize + Tidx;

        auto Idx = F.declVar<int>("Idx");
        auto K = F.declVar<int>("K");
        auto Zero = F.defRuntimeConst(0);
        auto One = F.defRuntimeConst(1);
        auto Nvar = F.defRuntimeConst(N);
        auto TileSizeConst = F.defRuntimeConst(TileSize);
        auto Accum = F.declVar<double>("accum");
        Accum = 0.0;

        F.forLoop({Idx, Zero, Nvar, TileSizeConst},
                    [&]() {
                        auto ACol = Idx + Tidx;
                        auto BRow = Idx + Tidy;
                        auto SIdx = Tidy * TileSize + Tidx;

                        auto AIdx = Row * N + ACol;
                        auto BIdx = BRow * N + Col;

                        AsTile[SIdx] = A[AIdx];
                        BsTile[SIdx] = B[BIdx];

                        F.callBuiltin(syncThreads);

                        F.forLoop({K, Zero, TileSizeConst, One},
                                            [&]() {
                                                auto AVal = AsTile[Tidy * TileSize + K];
                                                auto BVal = BsTile[K * TileSize + Tidx];
                                                Accum += AVal * BVal;
                                            });
                        F.callBuiltin(syncThreads);
                        K = 0;
                    });
        auto CIdx = Row * N + Col;
        C[CIdx] = Accum;

        F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

int main() {
  proteus::init();
  constexpr int N = 8192;
  constexpr int TileSize = 32;
  auto [JitMod, KernelHandle] = getTiledMatmulKernel(N, TileSize);

  JitMod->compile();

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

  gpuErrCheck(KernelHandle.launch({N / TileSize, N / TileSize, 1}, {TileSize, TileSize, 1}, 0, nullptr, CD, AD, BD));
  // hip_tiled_matmul_launch(A_d, B_d, C_d, N);
  // hip_nontiled_matmul_launch(A_d, B_d, C_d, N);
  gpuErrCheck(gpuDeviceSynchronize());

  // F(C, A, B);

//   // Timed trials
  const int NumTrials = 5;
  double TotalMs = 0.0;
  auto Start = std::chrono::high_resolution_clock::now();
  for (int T = 0; T < NumTrials; ++T) {
    gpuErrCheck(KernelHandle.launch({N / TileSize, N / TileSize, 1}, {TileSize, TileSize, 1}, 0, nullptr, CD, AD, BD));
    // hip_tiled_matmul_launch(A_d, B_d, C_d, N);
    // hip_nontiled_matmul_launch(A_d, B_d, C_d, N);
  }
  gpuErrCheck(gpuDeviceSynchronize());


  auto End = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> Ms = End - Start;
  TotalMs = Ms.count();


  double AvgMs = TotalMs / static_cast<double>(NumTrials);
  std::cerr.setf(std::ios::fixed);
  std::cerr.precision(3);
  std::cerr << "Average over " << NumTrials << " trials: " << AvgMs << " ms"
            << '\n';

  // Final result back to host after timing loop
  gpuErrCheck(gpuMemcpy(CH, CD, Bytes, gpuMemcpyDeviceToHost));
  for (int I = 0; I < N; I++) {
    for (int J = 0; J < N; J++) {
      if(CH[I*N + J] != N) {
        std::cout << "C_h[" << I << "][" << J << "] = " << CH[I*N + J] << '\n';
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
