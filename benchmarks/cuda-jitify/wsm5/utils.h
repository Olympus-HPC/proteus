// thread block sizes
#define XXX 8
#define YYY 8

// data allocation on host and device
// data initialization on host
// host to device copy
# define TODEV(A,s) A = (float*) malloc ((s) * sizeof(float)); \
                    for (int i = 0; i < s; i++) A[i] = 0.001; \
                    float *A##_d;\
                    cudaMalloc((void**)&A##_d,((s))*sizeof(float));\
                    cudaMemcpy(A##_d, A, (s)*sizeof(float), cudaMemcpyHostToDevice);

// device to host copy
# define FROMDEV(A,s) cudaMemcpy(A, A##_d, (s)*sizeof(float), cudaMemcpyDeviceToHost);

// deallocation host and device memory
# define FREE(A) free(A);\
                 cudaFree(A##_d)

# define TODEV3(A) TODEV(A,d3)
# define TODEV2(A) TODEV(A,d2)
# define FROMDEV3(A) FROMDEV(A,d3)
# define FROMDEV2(A) FROMDEV(A,d2)

