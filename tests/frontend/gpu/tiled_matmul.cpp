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

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif


// clang-format off
// No FileCheck for now; program prints the resulting C matrix

#if PROTEUS_ENABLE_HIP
#ifndef MATMUL_TILE_SIZE
#define MATMUL_TILE_SIZE 32
#endif

__global__ void hip_tiled_matmul_kernel(const double * __restrict__ A,
                                        const double * __restrict__ B,
                                        double * __restrict__ C,
                                        int N) {
  __shared__ double As[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
  __shared__ double Bs[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * MATMUL_TILE_SIZE + ty;
  int col = blockIdx.x * MATMUL_TILE_SIZE + tx;

  double sum = 0.0;
  int numTiles = (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE;

  for (int t = 0; t < numTiles; ++t) {
    int aCol = t * MATMUL_TILE_SIZE + tx;
    int bRow = t * MATMUL_TILE_SIZE + ty;

    As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
    Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < MATMUL_TILE_SIZE; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = sum;
  }
}

static inline void hip_tiled_matmul_launch(const double *A,
                                           const double *B,
                                           double *C,
                                           int N) {
  dim3 block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
  dim3 grid((N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
            (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
            1);
  hip_tiled_matmul_kernel<<<grid, block>>>(A, B, C, N);
}

// Non-tiled (naive) HIP kernel for matrix multiplication: C = A * B
__global__ void hip_nontiled_matmul_kernel(const double * __restrict__ A,
                                           const double * __restrict__ B,
                                           double * __restrict__ C,
                                           int N) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;

  if (row < N && col < N) {
    double sum = 0.0;
    #pragma unroll 1
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

static inline void hip_nontiled_matmul_launch(const double *A,
                                              const double *B,
                                              double *C,
                                              int N) {
  dim3 block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y,
            1);
  hip_nontiled_matmul_kernel<<<grid, block>>>(A, B, C, N);
}
#endif


auto getTiledMatmulKernel(int N, int TILE_SIZE) {
  auto JitMod = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      JitMod->addKernel<double *, double *, double *>("tiled_matmul");
    auto &F = KernelHandle.F;
  {

    auto Args = F.getArgs();
    auto &C = std::get<0>(Args);
    auto &A = std::get<1>(Args);
    auto &B = std::get<2>(Args);
    auto &AsTile = F.declArray<double>(TILE_SIZE * TILE_SIZE, Array::AddressSpace::SHARED);
    auto &BsTile = F.declArray<double>(TILE_SIZE * TILE_SIZE, Array::AddressSpace::SHARED);

    F.beginFunction();
    {

        auto Tidx = F.callBuiltin(getThreadIdX);
        auto Bidx = F.callBuiltin(getBlockIdX);
        auto Tidy = F.callBuiltin(getThreadIdY);
        auto Bidy = F.callBuiltin(getBlockIdY);

        auto Row = F.declVar<int>("Row");
        auto Col = F.declVar<int>("Col");

        Row = Bidy * TILE_SIZE + Tidy;
        Col = Bidx * TILE_SIZE + Tidx;

        auto Idx = F.declVar<int>("Idx");
        auto K = F.declVar<int>("K");
        auto Zero = F.defRuntimeConst(0);
        auto One = F.defRuntimeConst(1);
        auto Nvar = F.defRuntimeConst(N);
        auto TileSizeConst = F.defRuntimeConst(TILE_SIZE);
        auto accum = F.declVar<double>("accum");
        accum = 0.0;

        F.LoopNest({F.ForLoop({Idx, Zero, Nvar, TileSizeConst},
                    [&]() {
                        auto aCol = Idx + Tidx;
                        auto bRow = Idx + Tidy;
                        auto sIdx = Tidy * TILE_SIZE + Tidx;

                        auto AIdx = Row * N + aCol;
                        auto BIdx = bRow * N + Col;

                        AsTile[sIdx] = A[AIdx];
                        BsTile[sIdx] = B[BIdx];

                        F.callBuiltin(builtins::hip::syncThreads);

                        F.LoopNest({
                        F.ForLoop({K, Zero, TileSizeConst, One},
                                            [&]() {
                                                auto aVal = AsTile[Tidy * TILE_SIZE + K];
                                                auto bVal = BsTile[K * TILE_SIZE + Tidx];
                                                accum += aVal * bVal;
                                            })
                                          }).emit();
                        F.callBuiltin(builtins::hip::syncThreads);
                        K= 0;
                    })})
          .emit();
        auto CIdx = Row * N + Col;
        C[CIdx] = accum;

        F.ret();
    }
    F.endFunction();
  }
  return std::make_pair(std::move(JitMod), KernelHandle);
}

int main() {
  proteus::init();
  constexpr int N = 8192;
  constexpr int TILE_SIZE = 32;
  auto [JitMod, KernelHandle] = getTiledMatmulKernel(N, TILE_SIZE);

  JitMod->compile();

  // Host allocations
  double *A_h = (double *)new double[N * N];
  double *B_h = (double *)new double[N * N];
  double *C_h = (double *)new double[N * N];

  // Device allocations
  double *A_d; 
  double *B_d; 
  double *C_d; 
  size_t bytes = sizeof(double) * N * N;
  gpuErrCheck(gpuMalloc((void **)&A_d, bytes));
  gpuErrCheck(gpuMalloc((void **)&B_d, bytes));
  gpuErrCheck(gpuMalloc((void **)&C_d, bytes));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        A_h[i*N+j] = (1.0);
        B_h[i*N+j] = (1.0);
        C_h[i*N+j] = 0.0;
      }
    }

  // Stage inputs to device
  gpuErrCheck(gpuMemcpy(A_d, A_h, bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(B_d, B_h, bytes, gpuMemcpyHostToDevice));
  gpuErrCheck(gpuMemcpy(C_d, C_h, bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(KernelHandle.launch({N / TILE_SIZE, N / TILE_SIZE, 1}, {TILE_SIZE, TILE_SIZE, 1}, 0, nullptr, C_d, A_d, B_d));
  // hip_tiled_matmul_launch(A_d, B_d, C_d, N);
  // hip_nontiled_matmul_launch(A_d, B_d, C_d, N);
  gpuErrCheck(gpuDeviceSynchronize());

  // F(C, A, B);

//   // Timed trials
  const int num_trials = 5;
  double total_ms = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < num_trials; ++t) {
    gpuErrCheck(KernelHandle.launch({N / TILE_SIZE, N / TILE_SIZE, 1}, {TILE_SIZE, TILE_SIZE, 1}, 0, nullptr, C_d, A_d, B_d));
    // hip_tiled_matmul_launch(A_d, B_d, C_d, N);
    // hip_nontiled_matmul_launch(A_d, B_d, C_d, N);
  }
  gpuErrCheck(gpuDeviceSynchronize());


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms = end - start;
  total_ms = ms.count();


  double avg_ms = total_ms / static_cast<double>(num_trials);
  std::cerr.setf(std::ios::fixed);
  std::cerr.precision(3);
  std::cerr << "Average over " << num_trials << " trials: " << avg_ms << " ms"
            << '\n';

  // Final result back to host after timing loop
  gpuErrCheck(gpuMemcpy(C_h, C_d, bytes, gpuMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if(C_h[i*N + j] != N) {
        std::cout << "C_h[" << i << "][" << j << "] = " << C_h[i*N + j] << '\n';
      }
    }
  }

  // Cleanup device and host memory
  gpuErrCheck(gpuFree(A_d));
  gpuErrCheck(gpuFree(B_d));
  gpuErrCheck(gpuFree(C_d));
  delete[] A_h;
  delete[] B_h;
  delete[] C_h;

  proteus::finalize();
  return 0;
}
