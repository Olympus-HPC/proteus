# Usage

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

## The Proteus Frontend API
!!! warning "This is under development"
    The Frontend interface is currently under development and _will_ evolve to
    include additional features for specialization and a more streamlined
    syntax.
    However, the core concepts described here are expected to remain
    consistent, even as the implementation improves.

### Introduction

Proteus offers a *Frontend* API—an alternative C++ interface for generating,
managing, and executing LLVM IR modules at runtime via JIT compilation.
Supporting both CPU (host) and GPU (CUDA/HIP) targets, this API enables users to
dynamically construct code modules using a mini-language that closely resembles
natural C++ syntax.
This approach offers more flexibility than code annotations, allowing users to
generate JIT-compiled code with expanded possibilities for runtime
specialization and optimization.
(Note: runtime optimizations are still under development.)

### Description

This interface introduces a mini-language for expressing computation, with the
following core abstractions for programmatically generating JIT code:

- **Types**: A subset of the C++ type system, including fundamental types such
as `int`, `size_t`, `float`, `double`, and their pointer or reference forms.
- **Modules**: Analogous to translation units, modules encapsulate code and
organize it into functions.
- **Functions**: Units of computation within modules, containing variables and
statements.
- **Variables**: Typed entities supporting operations, such standard arithmetic
operations (`+`, `-`, `*`, `/`) and indexed access (`[]`) for pointers.
- **Statements**: Control flow and actions within functions, including:
    - `If` (conditional execution)
    - `For` (looping)
    - `Call` / `CallBuiltin` (function or built-in calls)
    - `Ret` (function return)

These abstractions are used to construct code at ahigh level, allowing users to
define functions and control flow in a way that resembles C++ code but
JIT-compiled at runtime.

### Syntax and semantics

Here is a concise example that demonstrates how to define a JIT Module using the
Proteus Frontend API to illustrate the syntax and semantics of the API:

```cpp
// Define a JIT module targeting CUDA.
auto J = proteus::JitModule("cuda");

// Add a kernel with the signature: void add_vectors(double *A, double *B, size_t N)
auto KernelHandle = J.addKernel<double *, double *, size_t>("add_vectors");
auto &F = KernelHandle.F;

// Begin the function body.
F.beginFunction();
{
    // Declare local variables and argument getters.
    auto &I = F.declVar<size_t>("I");
    auto &Inc = F.declVar<size_t>("Inc");
    auto &A = F.getArg(0); // Pointer to vector A
    auto &B = F.getArg(1); // Pointer to vector B
    auto &N = F.getArg(2); // Vector size

    // Compute the global thread index.
    I = F.callBuiltin(builtins::cuda::getBlockIdX) *
            F.callBuiltin(builtins::cuda::getBlockDimX) +
            F.callBuiltin(builtins::cuda::getThreadIdX);

    // Compute the stride (total number of threads).
    Inc = F.callBuiltin(builtins::cuda::getGridDimX) *
                F.callBuiltin(builtins::cuda::getBlockDimX);

    // Strided loop: each thread processes multiple elements.
    F.beginFor(I, I, N, Inc);
    {
        A[I] = A[I] + B[I];
    }
    F.endFor();

    F.ret();
}
F.endFunction();
```

This example demonstrates how to use the Proteus Frontend API to
programmatically construct a CUDA kernel for vector addition.
The API enables you to declare variables, access function arguments, use
built-in CUDA identifiers, and define control flow in a familiar way.

We start by creating a JIT module `J` targeting "cuda", which corresponds to an
NVIDIA GPU using CUDA abstractions.
The call to `J.addKernel()` defines a kernel with the specified argument types
(using template parameters), similar to a `__global__` function in CUDA.
This returns a handle to the kernel, which will later be used for launching.
The kernel function itself is accessed via `F`, which serves as the entry point
for code generation.

The frontend API uses `begin*` and `end*` methods to mark the start and end of
code blocks.
In this example, `F.beginFunction()` begins the function body, and it is good
practice to use braces to clearly indicate the scope of JIT-generated code.

Variables are declared using `F.declVar`, specifying the type as a template
parameter and providing a name as a string.
Assignment and arithmetic operations on these variables use familiar C++
operators (`=`, `+`, `-`, `*`, `/`, etc.), making the code intuitive to write.
Function arguments are accessed with `F.getArg`, which retrieves the argument by
its 0-based index and returns a reference for use as a variable.

To implement a typical GPU-strided loop, the variable `I` is initialized to the
global thread index, and `Inc` is set to the total number of threads.
Built-in CUDA identifiers such as `threadIdx.x`, `blockIdx.x`, `blockDim.x`, and
`gridDim.x` are accessed using `F.callBuiltin` with the corresponding names from
the `builtins::cuda` namespace.

The computation loop is defined using `F.beginFor`, which takes the loop index,
start value, end value, and increment.
The loop body is inside braces (good practice!), and performs element-wise
addition of the vectors `A` and `B`, storing the result in `A`.
The loop is closed with `F.endFor()`.
Finally, `F.ret()` marks the function's return, and `F.endFunction()` completes
the function definition.

#### JIT compilation and execution

After defining the module, you need to compile it and launch the kernel. The
following code example (which continues from the previous section) demonstrates
this process:

```cpp
// ... Continuing from above ...

// Allocate and initialize input vectors A and B, and specify their size N.
double *A = ...; // Pointer to vector A
double *B = ...; // Pointer to vector B
size_t N = ...;  // Number of elements in each vector

// Finalize and compile the JIT module. No further code can be added after this.
J.compile();

// Configure the CUDA kernel launch parameters.
constexpr unsigned ThreadsPerBlock = 256;
unsigned NumBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

// Launch the JIT-compiled kernel with the specified grid and block dimensions.
// Arguments: grid size, block size, shared memory size (0), stream (nullptr),
// kernel arguments (A, B, N).
KernelHandle.launch(
    {NumBlocks, 1, 1}, {ThreadsPerBlock, 1, 1}, 0, nullptr, A, B, N);

// Synchronize to ensure kernel execution is complete.
cudaDeviceSynchronize();
```

This is accomplished by invoking `compile()`, which _finalizes_ the module,
preventing any further code additions, and compiles it into an executable object
in memory.

The `KernelHandle` is used to launch the kernel with the specified grid and
block dimensions, and the rest of launching parameters (shared memory size and
stream, both set to defaults).
Then, arguments to the kernel follow.
The `launch` method is designed to closely resemble the CUDA kernel launch
syntax using triple chevrons (`<<<>>>`).
After launch, the program includes a call to `cudaDeviceSynchronize` to ensure
kernel operations are finished before proceeding with host computation.


That's it for now! You have successfully JIT-compiled and executed a kernel.
As a final note, simply changing the target from `"cuda"` to `"hip"` allows the
same code to run on an AMD GPU device.
Stay tuned for future updates, particularly around runtime specialization and
optimization!

!!! info
    This example is included as a test for the CPU:
    [add_vector.cpp](https://github.com/Olympus-HPC/proteus/blob/main/tests/frontend/gpu/add_vectors.cpp)
    and GPU:
    [add_vector.cpp](https://github.com/Olympus-HPC/proteus/blob/main/tests/frontend/gpu/add_vectors.cpp)
