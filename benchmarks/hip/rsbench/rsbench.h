#include "hip/hip_runtime.h"
#include <assert.h>
#include <chrono>
#include <float.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define PI 3.14159265359

// typedefs
typedef enum __hm { SMALL, LARGE, XL, XXL } HM_size;

#define HISTORY_BASED 1
#define EVENT_BASED 2

#define STARTING_SEED 1070
#define INITIALIZATION_SEED 42

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

typedef struct {
  double r;
  double i;
} RSComplex;

typedef struct {
  int nthreads;
  int n_nuclides;
  int lookups;
  HM_size HM;
  int avg_n_poles;
  int avg_n_windows;
  int numL;
  int doppler;
  int particles;
  int simulation_method;
  int kernel_id;
} Input;

typedef struct {
  RSComplex MP_EA;
  RSComplex MP_RT;
  RSComplex MP_RA;
  RSComplex MP_RF;
  short int l_value;
} Pole;

typedef struct {
  double T;
  double A;
  double F;
  int start;
  int end;
} Window;

typedef struct {
  int *n_poles;
  unsigned long length_n_poles;
  int *n_windows;
  unsigned long length_n_windows;
  Pole *poles;
  unsigned long length_poles;
  Window *windows;
  unsigned long length_windows;
  double *pseudo_K0RS;
  unsigned long length_pseudo_K0RS;
  int *num_nucs;
  unsigned long length_num_nucs;
  int *mats;
  unsigned long length_mats;
  double *concs;
  unsigned long length_concs;
  int max_num_nucs;
  int max_num_poles;
  int max_num_windows;
  double *p_energy_samples;
  unsigned long length_p_energy_samples;
  int *mat_samples;
  unsigned long length_mat_samples;
  unsigned long *verification;
  unsigned long length_verification;
} SimulationData;

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(int a);
Input read_CLI(int argc, char *argv[]);
void print_CLI_error(void);
void print_input_summary(Input input);
int validate_and_print_results(Input input, double runtime,
                               unsigned long vhash);

// init.c
SimulationData initialize_simulation(Input input);
int *generate_n_poles(Input input, uint64_t *seed);
int *generate_n_windows(Input input, uint64_t *seed);
Pole *generate_poles(Input input, int *n_poles, uint64_t *seed,
                     int *max_num_poles);
Window *generate_window_params(Input input, int *n_windows, int *n_poles,
                               uint64_t *seed, int *max_num_windows);
double *generate_pseudo_K0RS(Input input, uint64_t *seed);
SimulationData move_simulation_data_to_device(Input in, SimulationData SD);

// material.c
int *load_num_nucs(Input input);
int *load_mats(Input input, int *num_nucs, int *max_num_nucs,
               unsigned long *length_mats);
double *load_concs(int *num_nucs, uint64_t *seed, int max_num_nucs);
SimulationData get_materials(Input input, uint64_t *seed);

// utils.c
size_t get_mem_estimate(Input input);
double get_time(void);

// simulation.c
void run_event_based_simulation(Input input, SimulationData data,
                                unsigned long *vhash_result);
__global__ void xs_lookup_kernel_baseline(Input in, SimulationData GSD);
__global__ void xs_lookup_kernel_baseline(Input in, int max_num_nucs,
                                          int max_num_poles,
                                          int max_num_windows,
                                          SimulationData GSD);
__device__ void calculate_macro_xs(double *macro_xs, int mat, double E,
                                   Input input, int *num_nucs, int *mats,
                                   int max_num_nucs, double *concs,
                                   int *n_windows, double *pseudo_K0Rs,
                                   Window *windows, Pole *poles,
                                   int max_num_windows, int max_num_poles);
__device__ void calculate_micro_xs(double *micro_xs, int nuc, double E,
                                   Input input, int *n_windows,
                                   double *pseudo_K0RS, Window *windows,
                                   Pole *poles, int max_num_windows,
                                   int max_num_poles);
__device__ void calculate_micro_xs_doppler(double *micro_xs, int nuc, double E,
                                           Input input, int *n_windows,
                                           double *pseudo_K0RS, Window *windows,
                                           Pole *poles, int max_num_windows,
                                           int max_num_poles);
__device__ int pick_mat(uint64_t *seed);
__device__ void calculate_sig_T(int nuc, double E, Input input,
                                double *pseudo_K0RS, RSComplex *sigTfactors);
__device__ RSComplex fast_nuclear_W(RSComplex Z);
__host__ __device__ double LCG_random_double(uint64_t *seed);
__host__ __device__ uint64_t LCG_random_int(uint64_t *seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
__device__ RSComplex c_add(RSComplex A, RSComplex B);
__device__ RSComplex c_sub(RSComplex A, RSComplex B);
__host__ __device__ RSComplex c_mul(RSComplex A, RSComplex B);
__device__ RSComplex c_div(RSComplex A, RSComplex B);
__device__ double c_abs(RSComplex A);
__device__ double fast_exp(double x);
__device__ RSComplex fast_cexp(RSComplex z);
