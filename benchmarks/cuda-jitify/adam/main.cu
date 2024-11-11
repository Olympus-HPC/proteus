#include "../../external/jitify/jitify.hpp"
#include "reference.h"
#include <chrono>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const char *kernel_source =
    "the_kernel\n"
    "#include \"reference.h\"\n"
    "template<typename T, typename G, const int time_step, const size_t "
    "vector_size, const float b1>\n"
    "__global__\n"
    "void adam (\n"
    "        T* __restrict__ p,\n"
    "        T* __restrict__ m,\n"
    "        T* __restrict__ v,\n"
    "  const G* __restrict__ g,\n"
    "  const float b2,\n"
    "  const float eps,\n"
    "  const float grad_scale,\n"
    "  const float step_size,\n"
    "  adamMode_t mode,\n"
    "  const float decay)\n"
    "{\n"
    "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  const size_t totThreads = gridDim.x*blockDim.x;\n"
    "\n"
    "  for (size_t j = i; j < vector_size; j += totThreads) {\n"
    "    for (int t = 0; t < time_step; t++) {\n"
    "      T scaled_grad = g[j]/grad_scale;\n"
    "      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;\n"
    "      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;\n"
    "      float m_corrected = m[j] / (1.f-powf(b1, t));\n"
    "      float v_corrected = v[j] / (1.f-powf(b2, t));\n"
    "      float denom;\n"
    "      if (mode == ADAM_MODE_0)\n"
    "        denom = sqrtf(v_corrected + eps);\n"
    "      else // Mode 1\n"
    "        denom = sqrtf(v_corrected) + eps;\n"
    "      float update = (m_corrected/denom) + (decay*p[j]);\n"
    "      p[j] -= (step_size*update);\n"
    "    }\n"
    "  }\n"
    "}\n";

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n",
           argv[0]);
    return 1;
  }

  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(
      kernel_source, {}, {"-std=c++20", "-arch=sm_70", "--time=jitify.trace"});

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
  }

  float *d_m, *d_v, *d_g, *d_p;

  cudaMalloc((void **)&d_m, size_bytes);
  cudaMemcpy(d_m, m, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_v, size_bytes);
  cudaMemcpy(d_v, v, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_g, size_bytes);
  cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_p, size_bytes);
  cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice);

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

  cudaDeviceSynchronize();
  using jitify::reflection::Type;
  using jitify::reflection::type_of;

  auto start = std::chrono::steady_clock::now();
  auto configed = program.kernel("adam")
                      .instantiate(Type<float>{}, Type<float>{}, time_step,
                                   vector_size, beta1)
                      .configure(grids, blocks);

  for (int i = 0; i < repeat; i++) {
    configed.launch(d_p, d_m, d_v, d_g, beta2, eps, grad_scale, step_size, mode,
                    decay);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  cudaMemcpy(p, d_p, size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_p);
  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_g);

  // verify
  //  reference<float, float>(
  //    repeat,
  //    r, m, v, g,
  //    beta1, beta2,
  //    eps,
  //    grad_scale,
  //    step_size,
  //    time_step,
  //    vector_size,
  //    mode,
  //    decay);
  //
  //  bool ok = true;
  //  for (int i = 0; i < vector_size; i++) {
  //    if (r[i] - p[i] > 1e-3f) {
  //      ok = false;
  //      break;
  //    }
  //  }
  //  printf("%s\n", ok ? "PASS" : "FAIL");

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
