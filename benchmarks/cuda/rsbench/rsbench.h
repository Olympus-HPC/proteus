#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <cuda.h>

#define PI 3.14159265359

// typedefs
typedef enum __hm{SMALL, LARGE, XL, XXL} HM_size;

#define HISTORY_BASED 1
#define EVENT_BASED 2

#define STARTING_SEED 1070
#define INITIALIZATION_SEED 42

typedef struct{
	double r;
	double i;
} RSComplex;

typedef struct{
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

typedef struct{
	RSComplex MP_EA;
	RSComplex MP_RT;
	RSComplex MP_RA;
	RSComplex MP_RF;
	short int l_value;
} Pole;

typedef struct{
	double T;
	double A;
	double F;
	int start;
	int end;
} Window;

typedef struct{
	int * n_poles;
	unsigned long length_n_poles;
	int * n_windows;
	unsigned long length_n_windows;
	Pole * poles;
	unsigned long length_poles;
	Window * windows;
	unsigned long length_windows;
	double * pseudo_K0RS;
	unsigned long length_pseudo_K0RS;
	int * num_nucs;
	unsigned long length_num_nucs;
	int * mats;
	unsigned long length_mats;
	double * concs;
	unsigned long length_concs;
	int max_num_nucs;
	int max_num_poles;
	int max_num_windows;
	double * p_energy_samples;
	unsigned long length_p_energy_samples;
	int * mat_samples;
	unsigned long length_mat_samples;
} SimulationData;

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
Input read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_input_summary(Input input);
int validate_and_print_results(Input input, double runtime, unsigned long vhash, double kernel_init_time);

// init.c
SimulationData initialize_simulation( Input input );
int * generate_n_poles( Input input,  uint64_t * seed );
int * generate_n_windows( Input input ,  uint64_t * seed);
Pole * generate_poles( Input input, int * n_poles, uint64_t * seed, int * max_num_poles );
Window * generate_window_params( Input input, int * n_windows, int * n_poles, uint64_t * seed, int * max_num_windows );
double * generate_pseudo_K0RS( Input input, uint64_t * seed );

// material.c
int * load_num_nucs(Input input);
int * load_mats( Input input, int * num_nucs, int * max_num_nucs, unsigned long * length_mats );
double * load_concs( int * num_nucs, uint64_t * seed, int max_num_nucs );
SimulationData get_materials(Input input, uint64_t * seed);

// utils.c
size_t get_mem_estimate( Input input );
double get_time(void);

// simulation.c
__device__
RSComplex fast_cexp( RSComplex z );
__device__
RSComplex fast_nuclear_W( RSComplex Z );
template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_macro_xs( double * macro_xs, int mat, double E, int input_doppler, int input_numL, INT_T num_nucs, INT_T mats, int max_num_nucs, DOUBLE_T concs, INT_T n_windows, DOUBLE_T pseudo_K0Rs, WINDOW_T windows, POLE_T poles, int max_num_windows, int max_num_poles ) ;
template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, INT_T n_windows, DOUBLE_T pseudo_K0RS, WINDOW_T windows, POLE_T poles, int max_num_windows, int max_num_poles);
template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, INT_T n_windows, DOUBLE_T pseudo_K0RS, WINDOW_T windows, POLE_T poles, int max_num_windows, int max_num_poles );
template <class DOUBLE_T>
__device__
void calculate_sig_T( int nuc, double E, Input input, DOUBLE_T pseudo_K0RS, RSComplex * sigTfactors );
void run_event_based_simulation(Input in, SimulationData SD, unsigned long * vhash_result, double * kernel_init_time );
__host__ __device__
double LCG_random_double(uint64_t * seed);
uint64_t LCG_random_int(uint64_t * seed);
__device__
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
__device__
int pick_mat( uint64_t * seed );

// rscomplex.c
__device__
RSComplex c_add( RSComplex A, RSComplex B);
__device__
RSComplex c_sub( RSComplex A, RSComplex B);
__host__ __device__
RSComplex c_mul( RSComplex A, RSComplex B);
__device__
RSComplex c_div( RSComplex A, RSComplex B);
__device__
double c_abs( RSComplex A);

// papi.c
void counter_init( int *eventset, int *num_papi_events );
void counter_stop( int * eventset, int num_papi_events );
