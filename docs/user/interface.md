# User Interface

## Code Annotations

Proteus enables users to annotate (plain) functions or GPU kernels, and lambda
functions, for JIT compilation and optimization.
Users provide information to Proteus to specialize code for the runtime context
for generating more optimized code.

### Kernels / Functions

The interface for host functions or device kernel functions is the same.
Proteus leverage function attributes and specifically the `annotate` attribute,
which allows generic information to be attached on a function, be it a host
function or a kernel set to launch on a GPU device.

Let's highlight this annotations with an example:
```cpp title="Host"
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, size_t N, double *X, double *Y) {
    for(size_t I=0; I<N; ++I)
        Y[I] = A*X[I] + Y[I];
}
```
```cuda title="Device"
__global__
__attribute__((annotate("jit", 1, 2)))
void daxpy(double A, size_t N, double *X, double *Y) {
    for(size_t I=0; I<N; ++I)
        Y[I] = A*X[I] + Y[I];
}
```
This is DAXPY computation (double precision scaled vector addition).
The user annotates the `daxpy` function for JIT compilation using the `annotate`
attribute with the `jit` designator.
Further, the comma-separated list of numbers after `jit` corresponds to the
positions of arguments of the function (1-indexed) that Proteus JIT optimization should
specialize for.
In this case, those are the scaling factor `A` and the vector size `N`.

!!! note
    The argument list is **1-indexed**.

Proteus JIT compilation will fold designated arguments to constants when it
specializes the JIT module for this function.
The expectation that specialization will help the compiler optimization pipeline
(e.g., O3) to additionally optimize the code.
Folding values to constants enhnaces the scope and precision of compiler
analysis, which turbocharges optimizations such as constant propagation,
control-flow simplification, and unrolling.

### Lambdas

*Why Lambdas?*
The reason is that lambda functions are used extensively by portability
libraries, like RAJA and Kokkos, which are targets for Proteus JIT optimization.
Lambda functions are more complicated (in a sense) than plain functions since
they also capture calling context as closures.
Most often, the greatest opportunities for specialization for lambdas are in
their captured list, rather than their argument list.

Proteus  **requires** lambda function to be register with the Proteus runtime.
It provides a wrapper function, named `proteus::register_lambda` for this
purpose.

To specialize for a capture variable, Proteus **requires** uses to explicitly
initialize those variables using the wrapper function `proteus::jit_variable`.

Let's showcase the interfaces using an example:
```cpp title="Host"
auto DaxpyLambda = proteus::register_lambda(
    [=, A=proteus::jit_variable(A), N=proteus::jit_variable(N)]() {
    for(size_t I=0; I<N; ++I)
        Y[I] = A*X[I] + Y[I];
    }
);
DaxpyLambda(); 
```

```cuda title="Device"
template <typename Lambda>
__attribute__((annotate("jit"))) 
void kernel(Lambda &&Func) {
    Func();
}

auto DaxpyLambda = proteus::register_lambda(
    [=, A=proteus::jit_variable(A), N=proteus::jit_variable(N)] __device__ () {
    for(size_t I=0; I<N; ++I)
        Y[I] = A*X[I] + Y[I];
    }
);
kernel<<<blocks, threads>>>(DaxpyLambda);
```
!!! note
    It is necessary to denote the kernel that executes the Proteus-registered
    device lambda with the `jit` attribute.
