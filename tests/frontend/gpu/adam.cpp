// NOLINTBEGIN

// RUN: rm -rf .proteus
// RUN: ./adam.%ext 10000 200 100 | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>

#include "../../gpu/gpu_common.h"

#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace proteus;

#if PROTEUS_ENABLE_HIP

#define TARGET "hip"
#define getThreadIdX builtins::hip::getThreadIdX
#define getBlockIdX builtins::hip::getBlockIdX
#define getBlockDimX builtins::hip::getBlockDimX
#define getGridDimX builtins::hip::getGridDimX

#elif PROTEUS_ENABLE_CUDA

#define TARGET "cuda"
#define getThreadIdX builtins::cuda::getThreadIdX
#define getBlockIdX builtins::cuda::getBlockIdX
#define getBlockDimX builtins::cuda::getBlockDimX
#define getGridDimX builtins::cuda::getGridDimX

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

typedef enum {
  ADAM_MODE_0 = 0, // eps under square root
  ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;

template <typename T, typename G>
__global__ void
adam(T *__restrict__ p, T *__restrict__ m, T *__restrict__ v,
     const G *__restrict__ g, const float b1, const float b2, const float eps,
     const float grad_scale, const float step_size, const int time_step,
     const size_t vector_size, adamMode_t mode, const float decay) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t totThreads = gridDim.x * blockDim.x;

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j] / grad_scale;
      m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
      v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
      float m_corrected = m[j] / (1.f - powf(b1, t));
      float v_corrected = v[j] / (1.f - powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected / denom) + (decay * p[j]);
      p[j] -= (step_size * update);
    }
  }
}

std::unique_ptr<JitModule> createJitModule() {
  auto J = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      J->addKernel<float *, float *, float *, float *, float, float, float,
                   float, float, int, size_t, int, float>("adam");
  auto &F = KernelHandle.F;
  auto &p = F.getArg(0);
  auto &m = F.getArg(1);
  auto &v = F.getArg(2);
  auto &g = F.getArg(3);
  auto &b1 = F.getArg(4);
  auto &b2 = F.getArg(5);
  auto &eps = F.getArg(6);
  auto &grad_scale = F.getArg(7);
  auto &step_size = F.getArg(8);
  auto &time_step = F.getArg(9);
  auto &vector_size = F.getArg(10);
  auto &mode = F.getArg(11);
  auto &decay = F.getArg(12);

  auto &i = F.declVar<size_t>("i");
  auto &totThreads = F.declVar<size_t>("totThreads");
  auto &j = F.declVar<size_t>("j");
  auto &t = F.declVar<int>("t");
  auto &inc1 = F.declVar<int>("inc1");
  F.beginFunction();
  {
    i = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    totThreads = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    F.beginFor(j, i, vector_size, totThreads);
    {
      auto &lim = F.declVar<int>("lim");
      t = 1;
      inc1 = 1;
      lim = time_step + 1;
      F.beginFor(t, t, lim, inc1);
      {
        auto &scaled_grad = F.declVar<float>("scale_grad");
        scaled_grad = g[j] / grad_scale;

        m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;

        auto &m_corrected = F.declVar<float>("m_corrected");
        auto &v_corrected = F.declVar<float>("v_corrected");
        m_corrected = m[j] / (1.f - powf(b1, t));
        v_corrected = v[j] / (1.f - powf(b2, t));

        auto &denom = F.declVar<float>("denom");
        F.beginIf(mode == 0);
        { denom = sqrtf(v_corrected + eps); }
        F.endIf();

        F.beginIf(mode == 1);
        { denom = sqrtf(v_corrected) + eps; }
        F.endIf();

        auto &update = F.declVar<float>("update");
        update = (m_corrected / denom) + (decay * p[j]);

        p[j] -= (step_size * update);
      }
      F.endFor();
    }
    F.endFor();
    F.ret();
  }
  F.endFunction();

  return J;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n",
           argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  size_t size_bytes = vector_size * sizeof(float);

  float *m = (float *)malloc(size_bytes);
  float *v = (float *)malloc(size_bytes);
  float *g = (float *)malloc(size_bytes);
  float *p = (float *)malloc(size_bytes);
  float *r = (float *)malloc(size_bytes);

  srand(123);
  for (int i = 0; i < vector_size; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    r[i] = p[i] = rand() / (float)RAND_MAX;
    if (i < 10)
      std::cout << "init p[" << i << "] = " << p[i] << "\n";
  }

  float *d_m, *d_v, *d_g, *d_p;

  gpuErrCheck(gpuMalloc((void **)&d_m, size_bytes));
  gpuErrCheck(gpuMemcpy(d_m, m, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_v, size_bytes));
  gpuErrCheck(gpuMemcpy(d_v, v, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_g, size_bytes));
  gpuErrCheck(gpuMemcpy(d_g, g, size_bytes, gpuMemcpyHostToDevice));

  gpuErrCheck(gpuMalloc((void **)&d_p, size_bytes));
  gpuErrCheck(gpuMemcpy(d_p, p, size_bytes, gpuMemcpyHostToDevice));

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dim3 grids((vector_size + threadsPerBlock - 1) / threadsPerBlock);
  const dim3 blocks(threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "Creating JIT module\n";
  auto J = createJitModule();

  std::cout << "Compiling JIT module\n";
  J->compile();

  auto KernelHandle =
      J->getKernelHandle<float *, float *, float *, float *, float, float,
                         float, float, float, int, size_t, int, float>("adam");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // adam<float, float><<<grids, blocks>>>(d_p, d_m, d_v, d_g, beta1, beta2,
    // eps,
    //                                       grad_scale, step_size, time_step,
    //                                       vector_size, mode, decay);

    gpuErrCheck(KernelHandle.launch({grids.x, 1, 1}, {blocks.x, 1, 1}, 0, 0,
                                    d_p, d_m, d_v, d_g, beta1, beta2, eps,
                                    grad_scale, step_size, time_step,
                                    vector_size, mode, decay));
  }

  gpuErrCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  gpuErrCheck(gpuMemcpy(p, d_p, size_bytes, gpuMemcpyDeviceToHost));

  for (int j = 0; j < 10; ++j) {
    std::cout << "p[" << j << "] = " << p[j] << "\n";
    ;
  }

  gpuErrCheck(gpuFree(d_p));
  gpuErrCheck(gpuFree(d_m));
  gpuErrCheck(gpuFree(d_v));
  gpuErrCheck(gpuFree(d_g));

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}

// clang-format off
// We got slight differences in the output for the least significant digits.
// Could be HW, numeric, or the way we handle signedness.
// CHECK: init p[0] = 0.348563
// CHECK-NEXT: init p[1] = 0.259322
// CHECK-NEXT: init p[2] = 0.377145
// CHECK-NEXT: init p[3] = 0.486632
// CHECK-NEXT: init p[4] = 0.352038
// CHECK-NEXT: init p[5] = 0.0784863
// CHECK-NEXT: init p[6] = 0.968732
// CHECK-NEXT: init p[7] = 0.852707
// CHECK-NEXT: init p[8] = 0.153431
// CHECK-NEXT: init p[9] = 0.559506
// CHECK-NEXT: Creating JIT module
// CHECK-NEXT: Compiling JIT module
// CHECK-NEXT: Average kernel execution time {{.*}} (ms)
// CHECK-NEXT: p[0] = -0.57293
// CHECK-NEXT: p[1] = -0.59603
// CHECK-NEXT: p[2] = -0.592634
// CHECK-NEXT: p[3] = -0.588154
// CHECK-NEXT: p[4] = -0.593454
// CHECK-NEXT: p[5] = -0.59199
// CHECK-NEXT: p[6] = -0.573486
// CHECK-NEXT: p[7] = -0.599872
// CHECK-NEXT: p[8] = -0.58157
// CHECK-NEXT: p[9] = -0.59015{{[7|8]}}

// NOLINTEND
