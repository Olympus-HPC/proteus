#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

template <typename T> gpuError_t launcher(T kernel_in) {
  return gpuLaunchKernel((const void*)kernel_in, 1, 1, 0, 0, 0);
}

__global__ __attribute__((annotate("jit"))) void kernel_to_jit() {
  printf("Kernel to jit\n");
}

__global__  void kernel_no_jit() {
  printf("Kernel no jit\n");
}

int main() {
  gpuErrCheck(launcher(kernel_to_jit));
  gpuErrCheck(launcher(kernel_no_jit));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel to jit
// CHECK: Kernel no jit
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1