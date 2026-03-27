// pack_ambiguous.cpp
  #include <hip/hip_runtime.h>

  struct Pair {
    double a;
    double* p;
  };

  template <class F>
  __global__ void kern(long n, Pair s, F f) {
    if (threadIdx.x == 0) f(n, s);
  }

  int main() {
    double* p = nullptr;
    hipMalloc(&p, sizeof(double));

    Pair s{3.14, p};
    double a = s.a;

    // Closure layout is (double, double*) == Pair’s layout on typical ABIs.
    auto f = [=] __device__ (long i, Pair) {
      if (i) p[0] = a;
    };

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kern<decltype(f)>),
                       dim3(1), dim3(1), 0, 0,
                       7L, s, f);
    hipDeviceSynchronize();
  }