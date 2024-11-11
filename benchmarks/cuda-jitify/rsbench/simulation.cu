#include "../../external/jitify/jitify.hpp"
#include "rsbench.h"
#include "simulation.cuh"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, RSBench will only run the baseline implementation. Optimized
// variants must be specifically selected using the "-k <optimized variant ID>"
// command line argument.
////////////////////////////////////////////////////////////////////////////////////

void run_event_based_simulation(Input in, SimulationData SD,
                                unsigned long *vhash_result,
                                double *kernel_time) {
  printf("Beginning event based simulation...\n");

  // Let's create an extra verification array to reduce manually later on
  printf("Allocating an additional %.1lf MB of memory for verification "
         "arrays...\n",
         in.lookups * sizeof(int) / 1024.0 / 1024.0);
  int *verification_host = (int *)malloc(in.lookups * sizeof(int));

  // Scope here is important, as when we exit this blocl we will automatically
  // sync with device to ensure all work is done and that we can read from
  // verification_host array.

  // create a queue using the default device for the platform (cpu, gpu)

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printf("Running on: %s\n", devProp.name);
  printf("Initializing device buffers and JIT compiling kernel...\n");

  ////////////////////////////////////////////////////////////////////////////////
  // Create Device Buffers
  ////////////////////////////////////////////////////////////////////////////////
  int *verification_d = nullptr;
  int *mats_d = nullptr;
  int *num_nucs_d = nullptr;
  int *n_windows_d = nullptr;
  double *concs_d = nullptr;
  double *pseudo_K0RS_d = nullptr;
  Window *windows_d = nullptr;
  Pole *poles_d = nullptr;

  // assign SYCL buffer to existing memory
  // buffer<int, 1> num_nucs_d(SD.num_nucs,SD.length_num_nucs);
  cudaMalloc((void **)&num_nucs_d, sizeof(int) * SD.length_num_nucs);
  cudaMemcpy(num_nucs_d, SD.num_nucs, sizeof(int) * SD.length_num_nucs,
             cudaMemcpyHostToDevice);

  // buffer<double, 1> concs_d(SD.concs, SD.length_concs);
  cudaMalloc((void **)&concs_d, sizeof(double) * SD.length_concs);
  cudaMemcpy(concs_d, SD.concs, sizeof(double) * SD.length_concs,
             cudaMemcpyHostToDevice);

  // buffer<int, 1> mats_d(SD.mats, SD.length_mats);
  cudaMalloc((void **)&mats_d, sizeof(int) * SD.length_mats);
  cudaMemcpy(mats_d, SD.mats, sizeof(int) * SD.length_mats,
             cudaMemcpyHostToDevice);

  // buffer<int, 1> n_windows_d(SD.n_windows, SD.length_n_windows);
  cudaMalloc((void **)&n_windows_d, sizeof(int) * SD.length_n_windows);
  cudaMemcpy(n_windows_d, SD.n_windows, sizeof(int) * SD.length_n_windows,
             cudaMemcpyHostToDevice);

  // buffer<Pole, 1> poles_d(SD.poles, SD.length_poles);
  cudaMalloc((void **)&poles_d, sizeof(Pole) * SD.length_poles);
  cudaMemcpy(poles_d, SD.poles, sizeof(Pole) * SD.length_poles,
             cudaMemcpyHostToDevice);

  // buffer<Window, 1> windows_d(SD.windows, SD.length_windows);
  cudaMalloc((void **)&windows_d, sizeof(Window) * SD.length_windows);
  cudaMemcpy(windows_d, SD.windows, sizeof(Window) * SD.length_windows,
             cudaMemcpyHostToDevice);
  // buffer<double, 1> pseudo_K0RS_d(SD.pseudo_K0RS, SD.length_pseudo_K0RS);
  cudaMalloc((void **)&pseudo_K0RS_d, sizeof(double) * SD.length_pseudo_K0RS);
  cudaMemcpy(pseudo_K0RS_d, SD.pseudo_K0RS,
             sizeof(double) * SD.length_pseudo_K0RS, cudaMemcpyHostToDevice);

  // buffer<int, 1> verification_d(verification_host, in.lookups);
  cudaMalloc((void **)&verification_d, sizeof(int) * in.lookups);
  cudaMemcpy(verification_d, verification_host, sizeof(int) * in.lookups,
             cudaMemcpyHostToDevice);

  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(
      simulation_cu, {"rsbench.h"}, {"-std=c++20", "--time=jitify.trace"});

  double start = get_time();

  ////////////////////////////////////////////////////////////////////////////////
  // XS Lookup Simulation Loop
  ////////////////////////////////////////////////////////////////////////////////
  // lookup<<< dim3((in.lookups + 255) / 256), dim3(256) >>> (
  program.kernel("lookup")
      .instantiate(in.lookups, in.doppler, in.numL, SD.max_num_windows,
                   SD.max_num_poles, SD.max_num_nucs)
      .configure(((in.lookups + 255) / 256), 256)
      .launch(num_nucs_d, concs_d, mats_d, verification_d, n_windows_d,
              pseudo_K0RS_d, windows_d, poles_d);

  cudaDeviceSynchronize();
  double stop = get_time();
  printf(
      "Kernel initialization, compilation, and execution took %.2lf seconds.\n",
      stop - start);

  cudaMemcpy(verification_host, verification_d, sizeof(int) * in.lookups,
             cudaMemcpyDeviceToHost);

  cudaFree(verification_d);
  cudaFree(mats_d);
  cudaFree(num_nucs_d);
  cudaFree(concs_d);
  cudaFree(n_windows_d);
  cudaFree(windows_d);
  cudaFree(poles_d);
  cudaFree(pseudo_K0RS_d);

  // Host reduces the verification array
  unsigned long long verification_scalar = 0;
  for (int i = 0; i < in.lookups; i++)
    verification_scalar += verification_host[i];

  *vhash_result = verification_scalar;
  *kernel_time = stop - start;
}

double LCG_random_double(uint64_t *seed) {
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

uint64_t LCG_random_int(uint64_t *seed) {
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

RSComplex c_mul(RSComplex A, RSComplex B) {
  double a = A.r;
  double b = A.i;
  double c = B.r;
  double d = B.i;
  RSComplex C;
  C.r = (a * c) - (b * d);
  C.i = (a * d) + (b * c);
  return C;
}
