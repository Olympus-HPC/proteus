#include "rsbench.h"

size_t get_mem_estimate( Input input )
{
	size_t poles = input.n_nuclides * input.avg_n_poles * sizeof(Pole) + input.n_nuclides * sizeof(Pole *);
	size_t windows = input.n_nuclides * input.avg_n_windows * sizeof(Window) + input.n_nuclides * sizeof(Window *);
	size_t pseudo_K0RS = input.n_nuclides * input.numL * sizeof( double ) + input.n_nuclides * sizeof(double);
	size_t other = input.n_nuclides * 2 * sizeof(int);

	size_t total = poles + windows + pseudo_K0RS + other;
	
	return total;
}

double get_time(void)
{

	// If using C, we can do this:
	/* 
	struct timeval timecheck;
	gettimeofday(&timecheck, NULL);
	long ms = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
	double time = (double) ms / 1000.0;
	return time;
	*/
	
	// If using C++, we can do this:
	unsigned long us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1);
	return (double) us_since_epoch / 1.0e6;

}
