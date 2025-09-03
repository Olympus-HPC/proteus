#include "gpu_common.h"

template <typename T> struct Privatizer {
  using value_type = T;
  using reference_type = value_type &;
  value_type Priv;

  __host__ __device__ Privatizer(const T &o) : priv{o} {}

  __host__ __device__ reference_type get_priv() { return priv; }
};

template <typename T>
__host__ __device__ auto threadPrivatize(const T &item) -> Privatizer<T> {
  return Privatizer<T>{item};
}

template <typename Lambda>
__global__ __attribute__((annotate("jit", 2))) void globalWrapper(Lambda lam,
                                                                  size_t N) {
  auto privatizer = threadPrivatize(lam);
  auto &body = privatizer.get_priv();
  std::size_t idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < N) {
    body(idx);
  }
}

template <typename Lambda, typename... Args>
void forall(size_t N, Lambda &&lam, Args &&...args) {
  const std::size_t GridSize = (((N) + (256) - 1) / (256));
  auto func = reinterpret_cast<const void *>(&globalWrapper<Lambda>);
#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL(func, dim3(GridSize), dim3(256), 0, 0, lam, N);
#elif PROTEUS_ENABLE_CUDA
  void *AArgs[] = {&lam, &N};
  cudaLaunchKernel(func, dim3(GridSize), dim3(256), AArgs, 0, 0);
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
}
