struct kernel_body_t {
  __device__ void operator()() { printf("Kernel body\n"); }
};

const kernel_body_t kernel_body{};

template <typename LB>
__global__ __attribute__((annotate("jit"))) void kernel(LB lb) {
  lb();
}

template <typename T> gpuError_t __attribute__((always_inline)) launcher(T lb) {
  auto func = reinterpret_cast<const void *>(&kernel<T>);
  void *args[] = {(void *)&lb};
  return gpuLaunchKernel(func, 1, 1, args, 0, 0);
}
