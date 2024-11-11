#include "rsbench.h"

// Moves all required data structures to the GPU's memory space
SimulationData move_simulation_data_to_device( Input in, SimulationData SD )
{
	printf("Allocating and moving simulation data to GPU memory space...\n");

	size_t sz;
	size_t total_sz = 0;

	// Shallow copy of CPU simulation data to GPU simulation data
	SimulationData GSD = SD;

	// Move data to GPU memory space
	sz = GSD.length_num_nucs * sizeof(int);
	gpuErrchk( hipMalloc((void **) &GSD.num_nucs, sz) );
	gpuErrchk( hipMemcpy(GSD.num_nucs, SD.num_nucs, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_concs * sizeof(double);
	gpuErrchk( hipMalloc((void **) &GSD.concs, sz) );
	gpuErrchk( hipMemcpy(GSD.concs, SD.concs, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_mats * sizeof(int);
	gpuErrchk( hipMalloc((void **) &GSD.mats, sz) );
	gpuErrchk( hipMemcpy(GSD.mats, SD.mats, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_n_poles * sizeof(int);
	gpuErrchk( hipMalloc((void **) &GSD.n_poles, sz) );
	gpuErrchk( hipMemcpy(GSD.n_poles, SD.n_poles, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_n_windows * sizeof(int);
	gpuErrchk( hipMalloc((void **) &GSD.n_windows, sz) );
	gpuErrchk( hipMemcpy(GSD.n_windows, SD.n_windows, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_poles * sizeof(Pole);
	gpuErrchk( hipMalloc((void **) &GSD.poles, sz) );
	gpuErrchk( hipMemcpy(GSD.poles, SD.poles, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_windows * sizeof(Window);
	gpuErrchk( hipMalloc((void **) &GSD.windows, sz) );
	gpuErrchk( hipMemcpy(GSD.windows, SD.windows, sz, hipMemcpyHostToDevice) );
	total_sz += sz;

	sz = GSD.length_pseudo_K0RS * sizeof(double);
	gpuErrchk( hipMalloc((void **) &GSD.pseudo_K0RS, sz) );
	gpuErrchk( hipMemcpy(GSD.pseudo_K0RS, SD.pseudo_K0RS, sz, hipMemcpyHostToDevice) );
	total_sz += sz;
	
	// Allocate verification array on device. This structure is not needed on CPU, so we don't
	// have to copy anything over.
	sz = in.lookups * sizeof(unsigned long);
	gpuErrchk( hipMalloc((void **) &GSD.verification, sz) );
	total_sz += sz;
	GSD.length_verification = in.lookups;
	
	// Synchronize
	gpuErrchk( hipPeekAtLastError() );
	gpuErrchk( hipDeviceSynchronize() );
	
	printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0 );

	return GSD;

}

SimulationData initialize_simulation( Input input )
{
	uint64_t seed = INITIALIZATION_SEED;
	
	// Get material data
	printf("Loading Hoogenboom-Martin material data...\n");
	SimulationData SD = get_materials( input, &seed ); 
	
	// Allocate & fill energy grids
	printf("Generating resonance distributions...\n");
	SD.n_poles = generate_n_poles( input, &seed );
	SD.length_n_poles = input.n_nuclides;

	// Allocate & fill Window grids
	printf("Generating window distributions...\n");
	SD.n_windows = generate_n_windows( input, &seed );
	SD.length_n_windows = input.n_nuclides;

	// Prepare full resonance grid
	printf("Generating resonance parameter grid...\n");
	SD.poles = generate_poles( input, SD.n_poles, &seed, &SD.max_num_poles );
	SD.length_poles = input.n_nuclides * SD.max_num_poles;

	// Prepare full Window grid
	printf("Generating window parameter grid...\n");
	SD.windows = generate_window_params( input, SD.n_windows, SD.n_poles, &seed, &SD.max_num_windows);
	SD.length_windows = input.n_nuclides * SD.max_num_windows;

	// Prepare 0K Resonances
	printf("Generating 0K l_value data...\n");
	SD.pseudo_K0RS = generate_pseudo_K0RS( input, &seed );
	SD.length_pseudo_K0RS = input.n_nuclides * input.numL;

	return SD;
}

int * generate_n_poles( Input input, uint64_t * seed )
{
	int total_resonances = input.avg_n_poles * input.n_nuclides;

	int * R = (int *) malloc( input.n_nuclides * sizeof(int));
	
	// Ensure all nuclides have at least 1 resonance
	for( int i = 0; i < input.n_nuclides; i++ )
		R[i] = 1;

	// Sample the rest
	for( int i = 0; i < total_resonances - input.n_nuclides; i++ )
		R[LCG_random_int(seed) % input.n_nuclides]++;
	
	/* Debug	
	for( int i = 0; i < input.n_nuclides; i++ )
		printf("R[%d] = %d\n", i, R[i]);
	*/
	
	return R;
}

int * generate_n_windows( Input input, uint64_t * seed )
{
	int total_resonances = input.avg_n_windows * input.n_nuclides;

	int * R = (int *) malloc( input.n_nuclides * sizeof(int));
	
	// Ensure all nuclides have at least 1 resonance
	for( int i = 0; i < input.n_nuclides; i++ )
		R[i] = 1;

	// Sample the rest
	for( int i = 0; i < total_resonances - input.n_nuclides; i++ )
		R[LCG_random_int(seed) % input.n_nuclides]++;
	
	/* Debug	
	for( int i = 0; i < input.n_nuclides; i++ )
		printf("R[%d] = %d\n", i, R[i]);
	*/
	
	return R;
}

Pole * generate_poles( Input input, int * n_poles, uint64_t * seed, int * max_num_poles )
{
	// Pole Scaling Factor -- Used to bias hitting of the fast Faddeeva
	// region to approximately 99.5% (i.e., only 0.5% of lookups should
	// require the full eval).
	double f = 152.5;
	RSComplex f_c = {f, 0};

	int max_poles = -1;
	
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		if( n_poles[i] > max_poles)
			max_poles = n_poles[i];
	}
	*max_num_poles = max_poles;

	// Allocating 2D matrix as a 1D contiguous vector
	Pole * R = (Pole *) malloc( input.n_nuclides * max_poles * sizeof(Pole));
	
	// fill with data
	for( int i = 0; i < input.n_nuclides; i++ )
		for( int j = 0; j < n_poles[i]; j++ )
		{
			double r = LCG_random_double(seed);
			double im = LCG_random_double(seed);
			RSComplex t1 = {r, im};
			R[i * max_poles + j].MP_EA = c_mul(f_c,t1);
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t2 = {f*r, im};
			R[i * max_poles + j].MP_RT = t2;
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t3 = {f*r, im};
			R[i * max_poles + j].MP_RA = t3;
			r = LCG_random_double(seed);
			im = LCG_random_double(seed);
			RSComplex t4 = {f*r, im};
			R[i * max_poles + j].MP_RF = t4;
			R[i * max_poles + j].l_value = LCG_random_int(seed) % input.numL;
		}
	
	/* Debug
	for( int i = 0; i < input.n_nuclides; i++ )
		for( int j = 0; j < n_poles[i]; j++ )
			printf("R[%d][%d]: Eo = %lf lambda_o = %lf Tn = %lf Tg = %lf Tf = %lf\n", i, j, R[i * max_poles + j].Eo, R[i * max_poles + j].lambda_o, R[i * max_poles + j].Tn, R[i * max_poles + j].Tg, R[i * max_poles + j].Tf);
	*/

	return R;
}

Window * generate_window_params( Input input, int * n_windows, int * n_poles, uint64_t * seed, int * max_num_windows )
{
	int max_windows = -1;
	
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		if( n_windows[i] > max_windows)
			max_windows = n_windows[i];
	}
	*max_num_windows = max_windows;

	// Allocating 2D contiguous matrix
	Window * R = (Window *) malloc( input.n_nuclides * max_windows * sizeof(Window));
	
	// fill with data
	for( int i = 0; i < input.n_nuclides; i++ )
	{
		int space = n_poles[i] / n_windows[i];
		int remainder = n_poles[i] - space * n_windows[i];
		int ctr = 0;
		for( int j = 0; j < n_windows[i]; j++ )
		{
			R[i * max_windows + j].T = LCG_random_double(seed);
			R[i * max_windows + j].A = LCG_random_double(seed);
			R[i * max_windows + j].F = LCG_random_double(seed);
			R[i * max_windows + j].start = ctr; 
			R[i * max_windows + j].end = ctr + space - 1;

			ctr += space;

			if ( j < remainder )
			{
				ctr++;
				R[i * max_windows + j].end++;
			}
		}
	}

	return R;
}

double * generate_pseudo_K0RS( Input input, uint64_t * seed )
{
	double * R = (double *) malloc( input.n_nuclides * input.numL * sizeof(double));

	for( int i = 0; i < input.n_nuclides; i++)
		for( int j = 0; j < input.numL; j++ )
			R[i * input.numL + j] = LCG_random_double(seed);

	return R;
}
