#include "rsbench.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, RSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

__global__ 
#ifdef ENABLE_JIT
__attribute__((annotate("jit",9,10,11,12,13,14)))
#endif
void lookup ( 
    const int*__restrict__ num_nucs,
    const double*__restrict__ concs,
    const int*__restrict__ mats,
          int*__restrict__ verification,
    const int*__restrict__ n_windows,
    const double*__restrict__ pseudo_K0RS,
    const Window*__restrict__ windows,
    const Pole*__restrict__ poles,
    int n_lookups,
    int input_doppler,
    int input_numL,
    int max_num_windows,
    int max_num_poles,
    int max_num_nucs ) {

  // get the index to operate on, first dimemsion
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n_lookups) {

    // Set the initial seed value
    uint64_t seed = STARTING_SEED;  

    // Forward seed to lookup index (we need 2 samples per lookup)
    seed = fast_forward_LCG(seed, 2*i);

    // Randomly pick an energy and material for the particle
    double p_energy = LCG_random_double(&seed);
    int mat         = pick_mat(&seed); 

    // debugging
    //printf("E = %lf mat = %d\n", p_energy, mat);

    double macro_xs_vector[4] = {0};

    // Perform macroscopic Cross Section Lookup
    calculate_macro_xs(
        macro_xs_vector, mat, p_energy, 
        input_doppler, //in, 
        input_numL,
        num_nucs, mats, 
        max_num_nucs, concs, n_windows, pseudo_K0RS, windows, poles, 
        max_num_windows, max_num_poles );

    // For verification, and to prevent the compiler from optimizing
    // all work out, we interrogate the returned macro_xs_vector array
    // to find its maximum value index, then increment the verification
    // value by that index. In this implementation, we store to a global
    // array that will get tranferred back and reduced on the host.
    double max = -DBL_MAX;
    int max_idx = 0;
    for(int j = 0; j < 4; j++ )
    {
      if( macro_xs_vector[j] > max )
      {
        max = macro_xs_vector[j];
        max_idx = j;
      }
    }
    verification[i] = max_idx+1;
  }
}


void run_event_based_simulation(Input in, SimulationData SD, unsigned long * vhash_result, double * kernel_time )
{
  printf("Beginning event based simulation...\n");

  // Let's create an extra verification array to reduce manually later on
  printf("Allocating an additional %.1lf MB of memory for verification arrays...\n", in.lookups * sizeof(int) /1024.0/1024.0);
  int * verification_host = (int *) malloc(in.lookups * sizeof(int));

  // Scope here is important, as when we exit this blocl we will automatically sync with device
  // to ensure all work is done and that we can read from verification_host array.
  
  // create a queue using the default device for the platform (cpu, gpu)

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printf("Running on: %s\n", devProp.name);
  printf("Initializing device buffers and JIT compiling kernel...\n");

  ////////////////////////////////////////////////////////////////////////////////
  // Create Device Buffers
  ////////////////////////////////////////////////////////////////////////////////
  int *verification_d = nullptr;
  int *mats_d = nullptr ;
  int *num_nucs_d = nullptr;
  int *n_windows_d = nullptr;
  double *concs_d = nullptr;
  double *pseudo_K0RS_d = nullptr;
  Window *windows_d = nullptr;
  Pole *poles_d = nullptr;

  // assign SYCL buffer to existing memory
  //buffer<int, 1> num_nucs_d(SD.num_nucs,SD.length_num_nucs);
  cudaMalloc((void**)&num_nucs_d, sizeof(int) * SD.length_num_nucs);
  cudaMemcpy(num_nucs_d, SD.num_nucs, sizeof(int) * SD.length_num_nucs, cudaMemcpyHostToDevice);

  //buffer<double, 1> concs_d(SD.concs, SD.length_concs);
  cudaMalloc((void**)&concs_d, sizeof(double) * SD.length_concs);
  cudaMemcpy(concs_d, SD.concs, sizeof(double) * SD.length_concs, cudaMemcpyHostToDevice);

  //buffer<int, 1> mats_d(SD.mats, SD.length_mats);
  cudaMalloc((void**)&mats_d, sizeof(int) * SD.length_mats);
  cudaMemcpy(mats_d, SD.mats, sizeof(int) * SD.length_mats, cudaMemcpyHostToDevice);

  //buffer<int, 1> n_windows_d(SD.n_windows, SD.length_n_windows);
  cudaMalloc((void**)&n_windows_d, sizeof(int) * SD.length_n_windows);
  cudaMemcpy(n_windows_d, SD.n_windows, sizeof(int) * SD.length_n_windows, cudaMemcpyHostToDevice);

  //buffer<Pole, 1> poles_d(SD.poles, SD.length_poles);
  cudaMalloc((void**)&poles_d, sizeof(Pole) * SD.length_poles);
  cudaMemcpy(poles_d, SD.poles, sizeof(Pole) * SD.length_poles, cudaMemcpyHostToDevice);

  //buffer<Window, 1> windows_d(SD.windows, SD.length_windows);
  cudaMalloc((void**)&windows_d, sizeof(Window) * SD.length_windows);
  cudaMemcpy(windows_d, SD.windows, sizeof(Window) * SD.length_windows, cudaMemcpyHostToDevice);
  //buffer<double, 1> pseudo_K0RS_d(SD.pseudo_K0RS, SD.length_pseudo_K0RS);
  cudaMalloc((void**)&pseudo_K0RS_d, sizeof(double) * SD.length_pseudo_K0RS);
  cudaMemcpy(pseudo_K0RS_d, SD.pseudo_K0RS, sizeof(double) * SD.length_pseudo_K0RS, cudaMemcpyHostToDevice);

  //buffer<int, 1> verification_d(verification_host, in.lookups);
  cudaMalloc((void**)&verification_d, sizeof(int) * in.lookups);
  cudaMemcpy(verification_d, verification_host, sizeof(int) * in.lookups, cudaMemcpyHostToDevice);

  double start = get_time();

  ////////////////////////////////////////////////////////////////////////////////
  // XS Lookup Simulation Loop
  ////////////////////////////////////////////////////////////////////////////////
  lookup<<< dim3((in.lookups + 255) / 256), dim3(256) >>> (
      num_nucs_d, 
      concs_d, 
      mats_d, 
      verification_d, 
      n_windows_d, 
      pseudo_K0RS_d,
      windows_d,
      poles_d,
      in.lookups, 
      in.doppler,
      in.numL,
      SD.max_num_windows,
      SD.max_num_poles,
      SD.max_num_nucs );

  cudaDeviceSynchronize();
  double stop = get_time();
  printf("Kernel initialization, compilation, and execution took %.2lf seconds.\n", stop-start);

  cudaMemcpy(verification_host, verification_d, sizeof(int) * in.lookups, cudaMemcpyDeviceToHost);

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
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification_host[i];

  *vhash_result = verification_scalar;
  *kernel_time = stop-start;
}

template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_macro_xs(double * macro_xs, int mat, double E,
                        int input_doppler, int input_numL,
                        INT_T num_nucs, INT_T mats,
                        int max_num_nucs,
                        DOUBLE_T concs,
                        INT_T n_windows,
                        DOUBLE_T pseudo_K0Rs,
                        WINDOW_T windows,
                        POLE_T poles,
                        int max_num_windows,
                        int max_num_poles ) 
{
  // zero out macro vector
  for( int i = 0; i < 4; i++ )
    macro_xs[i] = 0;

  // for nuclide in mat
  for( int i = 0; i < num_nucs[mat]; i++ )
  {
    double micro_xs[4];
    int nuc = mats[mat * max_num_nucs + i];

    if( input_doppler == 1 )
      calculate_micro_xs_doppler( micro_xs, nuc, E, input_numL, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
    else
      calculate_micro_xs( micro_xs, nuc, E, input_numL, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

    for( int j = 0; j < 4; j++ )
    {
      macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
    }
    // Debug
    /*
       printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
       E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
     */
  }

  // Debug
  /*
     printf("E = %.2lf, mat = %d, macro_xs[0] = %.2lf, macro_xs[1] = %.2lf, macro_xs[2] = %.2lf, macro_xs[3] = %.2lf\n",
     E, mat, macro_xs[0], macro_xs[1], macro_xs[2], macro_xs[3] );
   */
}

// No Temperature dependence (i.e., 0K evaluation)
template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_micro_xs(double * micro_xs, int nuc, double E, int input_numL,
                        INT_T n_windows, DOUBLE_T pseudo_K0RS, WINDOW_T windows,
                        POLE_T poles, int max_num_windows, int max_num_poles)
{
  // MicroScopic XS's to Calculate
  double sigT;
  double sigA;
  double sigF;
  double sigE;

  // Calculate Window Index
  double spacing = 1.0 / n_windows[nuc];
  int window = (int) ( E / spacing );
  if( window == n_windows[nuc] )
    window--;

  // Calculate sigTfactors
  RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
  calculate_sig_T(nuc, E, input_numL, pseudo_K0RS, sigTfactors );

  // Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
  Window w = windows[nuc * max_num_windows + window];
  sigT = E * w.T;
  sigA = E * w.A;
  sigF = E * w.F;

  // Loop over Poles within window, add contributions
  for( int i = w.start; i < w.end; i++ )
  {
    RSComplex PSIIKI;
    RSComplex CDUM;
    Pole pole = poles[nuc * max_num_poles + i];
    RSComplex t1 = {0, 1};
    RSComplex t2 = {sqrt(E), 0 };
    PSIIKI = c_div( t1 , c_sub(pole.MP_EA,t2) );
    RSComplex E_c = {E, 0};
    CDUM = c_div(PSIIKI, E_c);
    sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;
    sigA += (c_mul( pole.MP_RA, CDUM)).r;
    sigF += (c_mul(pole.MP_RF, CDUM)).r;
  }

  sigE = sigT - sigA;

  micro_xs[0] = sigT;
  micro_xs[1] = sigA;
  micro_xs[2] = sigF;
  micro_xs[3] = sigE;
}

// Temperature Dependent Variation of Kernel
// (This involves using the Complex Faddeeva function to
// Doppler broaden the poles within the window)
template <class INT_T, class DOUBLE_T, class WINDOW_T, class POLE_T >
__device__
void calculate_micro_xs_doppler(double * micro_xs, int nuc, double E,
                                int input_numL, INT_T n_windows,
                                DOUBLE_T pseudo_K0RS, WINDOW_T windows,
                                POLE_T poles, int max_num_windows, int max_num_poles )
{
  // MicroScopic XS's to Calculate
  double sigT;
  double sigA;
  double sigF;
  double sigE;

  // Calculate Window Index
  double spacing = 1.0 / n_windows[nuc];
  int window = (int) ( E / spacing );
  if( window == n_windows[nuc] )
    window--;

  // Calculate sigTfactors
  RSComplex sigTfactors[4]; // Of length input.numL, which is always 4
  calculate_sig_T(nuc, E, input_numL, pseudo_K0RS, sigTfactors );

  // Calculate contributions from window "background" (i.e., poles outside window (pre-calculated)
  Window w = windows[nuc * max_num_windows + window];
  sigT = E * w.T;
  sigA = E * w.A;
  sigF = E * w.F;

  double dopp = 0.5;

  // Loop over Poles within window, add contributions
  for( int i = w.start; i < w.end; i++ )
  {
    Pole pole = poles[nuc * max_num_poles + i];

    // Prep Z
    RSComplex E_c = {E, 0};
    RSComplex dopp_c = {dopp, 0};
    RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);

    // Evaluate Fadeeva Function
    RSComplex faddeeva = fast_nuclear_W( Z );

    // Update W
    sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;
    sigA += (c_mul( pole.MP_RA , faddeeva)).r;
    sigF += (c_mul( pole.MP_RF , faddeeva)).r;
  }

  sigE = sigT - sigA;

  micro_xs[0] = sigT;
  micro_xs[1] = sigA;
  micro_xs[2] = sigF;
  micro_xs[3] = sigE;
}

// picks a material based on a probabilistic distribution
__device__
int pick_mat( uint64_t * seed )
{
  // I have a nice spreadsheet supporting these numbers. They are
  // the fractions (by volume) of material in the core. Not a 
  // *perfect* approximation of where XS lookups are going to occur,
  // but this will do a good job of biasing the system nonetheless.

  double dist[12];
  dist[0]  = 0.140;  // fuel
  dist[1]  = 0.052;  // cladding
  dist[2]  = 0.275;  // cold, borated water
  dist[3]  = 0.134;  // hot, borated water
  dist[4]  = 0.154;  // RPV
  dist[5]  = 0.064;  // Lower, radial reflector
  dist[6]  = 0.066;  // Upper reflector / top plate
  dist[7]  = 0.055;  // bottom plate
  dist[8]  = 0.008;  // bottom nozzle
  dist[9]  = 0.015;  // top nozzle
  dist[10] = 0.025;  // top of fuel assemblies
  dist[11] = 0.013;  // bottom of fuel assemblies

  double roll = LCG_random_double(seed);

  // makes a pick based on the distro
  for( int i = 0; i < 12; i++ )
  {
    double running = 0;
    for( int j = i; j > 0; j-- )
      running += dist[j];
    if( roll < running )
      return i;
  }

  return 0;
}

template <class DOUBLE_T>
__device__
void calculate_sig_T( int nuc, double E, int input_numL, DOUBLE_T pseudo_K0RS, RSComplex * sigTfactors )
{
  double phi;

  for( int i = 0; i < 4; i++ )
  {
    phi = pseudo_K0RS[nuc * input_numL + i] * sqrt(E);

    if( i == 1 )
      phi -= - atan( phi );
    else if( i == 2 )
      phi -= atan( 3.0 * phi / (3.0 - phi*phi));
    else if( i == 3 )
      phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));

    phi *= 2.0;

    sigTfactors[i].r = cos(phi);
    sigTfactors[i].i = -sin(phi);
  }
}

// This function uses a combination of the Abrarov Approximation
// and the QUICK_W three term asymptotic expansion.
// Only expected to use Abrarov ~0.5% of the time.
__device__
RSComplex fast_nuclear_W( RSComplex Z )
{
  // Abrarov 
  if( c_abs(Z) < 6.0 )
  {
    // Precomputed parts for speeding things up
    // (N = 10, Tm = 12.0)
    RSComplex prefactor = {0, 8.124330e+01};
    double an[10] = {
      2.758402e-01,
      2.245740e-01,
      1.594149e-01,
      9.866577e-02,
      5.324414e-02,
      2.505215e-02,
      1.027747e-02,
      3.676164e-03,
      1.146494e-03,
      3.117570e-04
    };
    double neg_1n[10] = {
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0
    };

    double denominator_left[10] = {
      9.869604e+00,
      3.947842e+01,
      8.882644e+01,
      1.579137e+02,
      2.467401e+02,
      3.553058e+02,
      4.836106e+02,
      6.316547e+02,
      7.994380e+02,
      9.869604e+02
    };

    RSComplex t1 = {0, 12};
    RSComplex t2 = {12, 0};
    RSComplex i = {0,1};
    RSComplex one = {1, 0};
    RSComplex W = c_div(c_mul(i, ( c_sub(one, fast_cexp(c_mul(t1, Z))) )) , c_mul(t2, Z));
    RSComplex sum = {0,0};
    for( int n = 0; n < 10; n++ )
    {
      RSComplex t3 = {neg_1n[n], 0};
      RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1, Z))), one);
      RSComplex t4 = {denominator_left[n], 0};
      RSComplex t5 = {144, 0};
      RSComplex bot = c_sub(t4, c_mul(t5,c_mul(Z,Z)));
      RSComplex t6 = {an[n], 0};
      sum = c_add(sum, c_mul(t6, c_div(top,bot)));
    }
    W = c_add(W, c_mul(prefactor, c_mul(Z, sum)));
    return W;
  }
  else
  {
    // QUICK_2 3 Term Asymptotic Expansion (Accurate to O(1e-6)).
    // Pre-computed parameters
    RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182,0};
    RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0};
    RSComplex c = {0.051765358792987823963876628425793170829107067780337219430904, 0};
    RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0};

    RSComplex i = {0,1};
    RSComplex Z2 = c_mul(Z, Z);
    // Three Term Asymptotic Expansion
    RSComplex W = c_mul(c_mul(Z,i), (c_add(c_div(a,(c_sub(Z2, b))) , c_div(c,(c_sub(Z2, d))))));

    return W;
  }
}

__host__ __device__
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

uint64_t LCG_random_int(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}  

__device__
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;

  n = n % m;

  uint64_t a_new = 1;
  uint64_t c_new = 0;

  while(n > 0) 
  {
    if(n & 1)
    {
      a_new *= a;
      c_new = c_new * a + c;
    }
    c *= (a + 1);
    a *= a;

    n >>= 1;
  }

  return (a_new * seed + c_new) % m;
}

// Complex arithmetic functions

__device__
RSComplex c_add( RSComplex A, RSComplex B)
{
  RSComplex C;
  C.r = A.r + B.r;
  C.i = A.i + B.i;
  return C;
}

__device__
RSComplex c_sub( RSComplex A, RSComplex B)
{
  RSComplex C;
  C.r = A.r - B.r;
  C.i = A.i - B.i;
  return C;
}

__host__ __device__
RSComplex c_mul( RSComplex A, RSComplex B)
{
  double a = A.r;
  double b = A.i;
  double c = B.r;
  double d = B.i;
  RSComplex C;
  C.r = (a*c) - (b*d);
  C.i = (a*d) + (b*c);
  return C;
}

__device__
RSComplex c_div( RSComplex A, RSComplex B)
{
  double a = A.r;
  double b = A.i;
  double c = B.r;
  double d = B.i;
  RSComplex C;
  double denom = c*c + d*d;
  C.r = ( (a*c) + (b*d) ) / denom;
  C.i = ( (b*c) - (a*d) ) / denom;
  return C;
}

__device__
double c_abs( RSComplex A)
{
  return sqrt(A.r*A.r + A.i * A.i);
}


// Fast (but inaccurate) exponential function
// Written By "ACMer":
// https://codingforspeed.com/using-faster-exponential-approximation/
// We use our own to avoid small differences in compiler specific
// exp() intrinsic implementations that make it difficult to verify
// if the code is working correctly or not.
__device__
double fast_exp(double x)
{
  x = 1.0 + x * 0.000244140625;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

// Implementation based on:
// z = x + iy
// cexp(z) = e^x * (cos(y) + i * sin(y))
__device__
RSComplex fast_cexp( RSComplex z )
{
  double x = z.r;
  double y = z.i;

  // For consistency across architectures, we
  // will use our own exponetial implementation
  //double t1 = exp(x);
  double t1 = fast_exp(x);
  double t2 = cos(y);
  double t3 = sin(y);
  RSComplex t4 = {t2, t3};
  RSComplex t5 = {t1, 0};
  RSComplex result = c_mul(t5, (t4));
  return result;
}  
