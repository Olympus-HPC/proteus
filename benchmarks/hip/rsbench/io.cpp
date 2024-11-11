#include "rsbench.h"

// Prints program logo
void logo(int version)
{
	border_print();
	printf(
"                    _____   _____ ____                  _     \n"
"                   |  __ \\ / ____|  _ \\                | |    \n"
"                   | |__) | (___ | |_) | ___ _ __   ___| |__  \n"
"                   |  _  / \\___ \\|  _ < / _ \\ '_ \\ / __| '_ \\ \n"
"                   | | \\ \\ ____) | |_) |  __/ | | | (__| | | |\n"
"                   |_|  \\_\\_____/|____/ \\___|_| |_|\\___|_| |_|\n\n"
	);
	border_print();
	center_print("Developed at Argonne National Laboratory", 79);
	char v[100];
	sprintf(v, "Version: %d", version);
	center_print(v, 79);
	border_print();
}

// Prints Section titles in center of 80 char terminal
void center_print(const char *s, int width)
{
	int length = strlen(s);
	int i;
	for (i=0; i<=(width-length)/2; i++) {
		fputs(" ", stdout);
	}
	fputs(s, stdout);
	fputs("\n", stdout);
}

void border_print(void)
{
	printf(
	"==================================================================="
	"=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( int a )
{
    if( a < 1000 )
        printf("%d\n",a);

    else if( a >= 1000 && a < 1000000 )
        printf("%d,%03d\n", a / 1000, a % 1000);

    else if( a >= 1000000 && a < 1000000000 )
        printf("%d,%03d,%03d\n", a / 1000000, (a % 1000000) / 1000, a % 1000 );

    else if( a >= 1000000000 )
        printf("%d,%03d,%03d,%03d\n",
               a / 1000000000,
               (a % 1000000000) / 1000000,
               (a % 1000000) / 1000,
               a % 1000 );
    else
        printf("%d\n",a);
}

Input read_CLI( int argc, char * argv[] )
{
	Input input;

	// defaults to the history based simulation method
	input.simulation_method = HISTORY_BASED;
	// defaults to max threads on the system	
	input.nthreads = 1;
	// defaults to 355 (corresponding to H-M Large benchmark)
	input.n_nuclides = 355;
	// defaults to 300,000
	input.particles = 300000;
	// defaults to 34
	input.lookups = 34;
	// defaults to H-M Large benchmark
	input.HM = LARGE;
	// defaults to 3000 resonancs (avg) per nuclide
	input.avg_n_poles = 1000;
	// defaults to 100
	input.avg_n_windows = 100;
	// defaults to 4;
	input.numL = 4;
	// defaults to no temperature dependence (Doppler broadening)
	input.doppler = 1;
	// defaults to baseline simulation kernel
	input.kernel_id = 0;
	
	int default_lookups = 1;
	int default_particles = 1;

	// Collect Raw Input
	for( int i = 1; i < argc; i++ )
	{
		char * arg = argv[i];

		// Simulation Method (-m)
		if( strcmp(arg, "-m") == 0 )
		{
			char * sim_type = NULL;
			if( ++i < argc )
				sim_type = argv[i];
			else
				print_CLI_error();

			if( strcmp(sim_type, "history") == 0 )
				input.simulation_method = HISTORY_BASED;
			else if( strcmp(sim_type, "event") == 0 )
			{
				input.simulation_method = EVENT_BASED;
				// Also resets default # of lookups
				if( default_lookups && default_particles )
				{
					input.lookups =  input.lookups * input.particles;
					input.particles = 0;
				}
			}
			else
				print_CLI_error();
		}
		// lookups (-l)
		else if( strcmp(arg, "-l") == 0 )
		{
			if( ++i < argc )
			{
				input.lookups = atoi(argv[i]);
				default_lookups = 0;
			}
			else
				print_CLI_error();
		}
		// particles (-p)
		else if( strcmp(arg, "-p") == 0 )
		{
			if( ++i < argc )
			{
				input.particles = atoi(argv[i]);
				default_particles = 0;
			}
			else
				print_CLI_error();
		}
		// nuclides (-n)
		else if( strcmp(arg, "-n") == 0 )
		{
			if( ++i < argc )
				input.n_nuclides = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// HM (-s)
		else if( strcmp(arg, "-s") == 0 )
		{	
			if( ++i < argc )
			{
				if( strcmp(argv[i], "small") == 0 )
					input.HM = SMALL;
				else if ( strcmp(argv[i], "large") == 0 )
					input.HM = LARGE;
				else
					print_CLI_error();
			}
			else
				print_CLI_error();
		}
		// Doppler Broadening (Temperature Dependence)
		else if( strcmp(arg, "-d") == 0 )
		{	
			input.doppler = 0;
		}
		// Avg number of windows per nuclide (-w)
		else if( strcmp(arg, "-W") == 0 )
		{
			if( ++i < argc )
				input.avg_n_windows = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// Avg number of poles per nuclide (-p)
		else if( strcmp(arg, "-P") == 0 )
		{
			if( ++i < argc )
				input.avg_n_poles = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// Kernel ID (-k)
		else if( strcmp(arg, "-k") == 0 )
		{
			if( ++i < argc )
				input.kernel_id = atoi(argv[i]);
			else
				print_CLI_error();
		}
		else
			print_CLI_error();
	}

	// Validate Input

	// Validate nthreads
	if( input.nthreads < 1 )
		print_CLI_error();
	
	// Validate n_isotopes
	if( input.n_nuclides < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.lookups < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.avg_n_poles < 1 )
		print_CLI_error();
	
	// Validate lookups
	if( input.avg_n_windows < 1 )
		print_CLI_error();
	
	// Set HM size specific parameters
	// (defaults to large)
	if( input.HM == SMALL )
		input.n_nuclides = 68;

	// Return input struct
	return input;
}

void print_CLI_error(void)
{
	printf("Usage: ./multibench <options>\n");
	printf("Options include:\n");
	printf("  -s <size>        Size of H-M Benchmark to run (small, large)\n");
	printf("  -l <lookups>     Number of Cross-section (XS) lookups per particle history\n");
	printf("  -p <particles>   Number of particle histories\n");
	printf("  -P <poles>       Average Number of Poles per Nuclide\n");
	printf("  -W <poles>       Average Number of Windows per Nuclide\n");
	printf("  -d               Disables Temperature Dependence (Doppler Broadening)\n");
	printf("Default is equivalent to: -s large -l 34 -p 300000 -P 1000 -W 100\n");
	printf("See readme for full description of default run values\n");
	exit(4);
}

void print_input_summary(Input input)
{
	printf("Programming Model:           HIP\n");
	// Print CUDA device name
	hipDeviceProp_t prop;
	int device;
	hipGetDevice(&device);
	hipGetDeviceProperties ( &prop, device );
	printf("HIP Device:                  %s\n", prop.name); 

	// Calculate Estimate of Memory Usage
	size_t mem = get_mem_estimate(input);

	if( input.simulation_method == EVENT_BASED )
		printf("Simulation Method:           Event Based\n");
	else
		printf("Simulation Method:           History Based\n");
	printf("Materials:                   12\n");
	printf("H-M Benchmark Size:          ");
	if( input.HM == 0 )
		printf("Small\n");
	else
		printf("Large\n");
	if( input.doppler == 1 )
		printf("Temperature Dependence:      ON\n");
	else
		printf("Temperature Dependence:      OFF\n");
	printf("Total Nuclides:              %d\n", input.n_nuclides);
	printf("Avg Poles per Nuclide:       "); fancy_int(input.avg_n_poles);
	printf("Avg Windows per Nuclide:     "); fancy_int(input.avg_n_windows);

	int lookups = input.lookups;
	if( input.simulation_method == HISTORY_BASED )
	{
		printf("Particles:                   "); fancy_int(input.particles);
		printf("XS Lookups per Particle:     "); fancy_int(input.lookups);
		lookups *= input.particles;
	}
	printf("Total XS Lookups:            "); fancy_int(lookups);
	printf("Est. Memory Usage (MB):      %.1lf\n", mem / 1024.0 / 1024.0);
}

int validate_and_print_results(Input input, double runtime, unsigned long vhash)
{
  printf("NOTE: Timings are estimated -- use nvprof/nsys/iprof/rocprof for formal analysis\n");
	printf("Runtime:               %.3lf seconds\n", runtime);
	int lookups = 0;
	if( input.simulation_method == HISTORY_BASED )
		lookups = input.lookups*input.particles;
	else
		lookups = input.lookups;
	printf("Lookups:               "); fancy_int(lookups);
	printf("Lookups/s:             "); fancy_int((double) lookups / (runtime));

	int is_invalid = 1;

	unsigned long long large = 0;
	unsigned long long small = 0;
	if(input.simulation_method == HISTORY_BASED )
	{
		large = 351485;
		small = 879693;
	}
	else if( input.simulation_method == EVENT_BASED )
	{
		large = 358389;
		small = 880018;
	}

	if( input.HM  == LARGE )
	{
		if( vhash == large )
		{
			printf("Verification checksum: %lu (Valid)\n", vhash);
			is_invalid = 0;
		}
		else
			printf("Verification checksum: %lu (WARNING - INAVALID CHECKSUM!)\n", vhash);
	}
	else if( input.HM  == SMALL )
	{
		if( vhash == small )
		{
			printf("Verification checksum: %lu (Valid)\n", vhash);
			is_invalid = 0;
		}
		else
			printf("Verification checksum: %lu (WARNING - INAVALID CHECKSUM!)\n", vhash);
	}

	return is_invalid;
}
