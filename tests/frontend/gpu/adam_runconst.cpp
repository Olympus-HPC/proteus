// NOLINTBEGIN

// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/adam_runconst.%ext 10000 200 100 | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/adam_runconst.%ext 10000 200 100 | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace proteus;
using namespace builtins::gpu;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
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

auto createJitModuleSpecial(float _b1, float _b2, float _eps, float _grad_scale,
                            float _step_size, int _time_step,
                            size_t _vector_size, int _mode, float _decay) {
  auto J = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      J->addKernel<void(float *, float *, float *, float *)>("adam");
  auto &F = KernelHandle.F;
  auto [p, m, v, g] = F.getArgs();

  auto i = F.declVar<size_t>("i");
  auto totThreads = F.declVar<size_t>("totThreads");
  auto j = F.declVar<size_t>("j");
  auto t = F.declVar<int>("t");
  auto inc1 = F.declVar<int>("inc1");

  F.beginFunction();
  {
    auto [b1, b2, eps, grad_scale, step_size, time_step, vector_size, mode,
          decay] = F.defRuntimeConsts(_b1, _b2, _eps, _grad_scale, _step_size,
                                      _time_step, _vector_size, _mode, _decay);

    i = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    totThreads = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    F.beginFor(j, i, vector_size, totThreads);
    {
      auto lim = F.declVar<int>("lim");
      t = 1;
      inc1 = 1;
      lim = time_step + 1;
      F.beginFor(t, t, lim, inc1);
      {
        auto scaled_grad = F.declVar<float>("scale_grad");
        scaled_grad = g[j] / grad_scale;

        m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;

        auto m_corrected = F.declVar<float>("m_corrected");
        auto v_corrected = F.declVar<float>("v_corrected");
        m_corrected = m[j] / (1.f - powf(b1, t));
        v_corrected = v[j] / (1.f - powf(b2, t));

        auto denom = F.declVar<float>("denom");
        F.beginIf(mode == 0);
        { denom = sqrtf(v_corrected + eps); }
        F.endIf();

        F.beginIf(mode == 1);
        { denom = sqrtf(v_corrected) + eps; }
        F.endIf();

        auto update = F.declVar<float>("update");
        update = (m_corrected / denom) + (decay * p[j]);

        p[j] -= (step_size * update);
      }
      F.endFor();
    }
    F.endFor();
    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n",
           argv[0]);
    return 1;
  }

  proteus::init();

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
  auto [J, KernelHandle] =
      createJitModuleSpecial(beta1, beta2, eps, grad_scale, step_size,
                             time_step, vector_size, mode, decay);

  std::cout << "Compiling JIT module\n";
  J->compile();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // adam<float, float><<<grids, blocks>>>(d_p, d_m, d_v, d_g, beta1, beta2,
    // eps,
    //                                       grad_scale, step_size, time_step,
    //                                       vector_size, mode, decay);

    gpuErrCheck(KernelHandle.launch({grids.x, 1, 1}, {blocks.x, 1, 1}, 0, 0,
                                    d_p, d_m, d_v, d_g));
  }

  gpuErrCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  gpuErrCheck(gpuMemcpy(p, d_p, size_bytes, gpuMemcpyDeviceToHost));

  for (int j = 0; j < 10; ++j) {
    std::cout << "p[" << j << "] = " << p[j] << "\n";
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
  proteus::finalize();
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
// CHECK-NEXT: p[5] = -0.591989
// CHECK-NEXT: p[6] = -0.573486
// CHECK-NEXT: p[7] = -0.599872
// CHECK-NEXT: p[8] = -0.58157
// CHECK-NEXT: p[9] = -0.59015{{[7|8]}}
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1

// NOLINTEND
