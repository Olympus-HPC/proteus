#include <cstdlib>
#include <cstddef>
#include <iostream>

__attribute__((annotate("jit", 1, 4)))
void my_daxpy(double a, double* x, double* y, int N)
{
    for (std::size_t i{0}; i < N; i++) {
        y[i] += x[i]*a; 
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    double* x = static_cast<double*>(malloc(sizeof(double)*N));
    double* y = static_cast<double*>(malloc(sizeof(double)*N));

    for (std::size_t i{0}; i < N; i++) {
        x[i] = 0.31414*i;
        y[i] = 0.0;
    }

    std::cout << y[10] << std::endl;
    my_daxpy(6.2, x, y, N);
    std::cout << y[10] << std::endl;
    my_daxpy(1.0, x, y, N);
    std::cout << y[10] << std::endl;


    free(x);
    free(y);
}