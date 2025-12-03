# Usage

## Code Annotations

Proteus lets you annotate plain functions, GPU kernels, and even lambdas for JIT
compilation and optimization.
By marking which arguments to specialize, Proteus can fold runtime values into
constants and generate more optimized code.

### Kernels / Functions

The interface for host functions and device kernel functions is the same.
Proteus relies on function attributes —specifically the `annotate` attribute —
which allows metadata to be attached to any function, whether it runs on the CPU
or as a GPU kernel.

Here’s a DAXPY (double-precision scaled vector addition) example:

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
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int stride = blockDim.x * gridDim.x;
    for(int i=tid; i<N; i+=stride)
      X[i] = A*X[i] + Y[i];
}
```

Here, the `annotate("jit", 1, 2)` attribute marks the function for JIT
compilation.
The comma-separated list after `jit` specifies which arguments
(1-indexed) Proteus should specialize: in this case, the scaling factor `A` and
the vector size `N`.

!!! note
    Function arguments are **1-indexed**.

When specializing, Proteus folds the designated arguments into constants.
This allows the downstream compiler optimization pipeline (e.g. at `-O3`) to
apply stronger transformations.
Folding runtime values into constants enhances analysis precision and unlocks
optimizations such as constant propagation, control-flow simplification, and
loop unrolling.

### Lambdas

Why lambdas?
Because portability libraries like RAJA and Kokkos make heavy use of them—and
these are prime targets for Proteus JIT optimization.

Lambdas are slightly more complex than plain functions because they capture
variables in closures.
In practice, the greatest opportunities for specialization are often in the
captured variables rather than the argument list.

Proteus therefore requires lambda functions to be **registered** with the runtime
using `proteus::register_lambda`.
To mark captured variables for specialization, they must be explicitly wrapped
with `proteus::jit_variable`.

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
    For device lambdas, the kernel that executes the Proteus-registered lambda
    must also be annotated with `jit`.


### C++ Annotations API

Beyond attribute-based annotations, Proteus also provides a more powerful C++
API for specialization through annotation functions.
This API supports specialization of **scalars**, **arrays**, and **objects**.

When using these APIs, the contract is that annotated data containers are
treated as **read-only** within the specialized code.

The C++ annotations API includes:

- `template<typename T> proteus::jit_arg(T Scalar)` — equivalent to the
  attribute-based annotation for specializing a scalar argument.
- `template<typename T> proteus::jit_array(T *Arr, size_t NumElts)` — specialize
  an array.
- `template<typename T> proteus::jit_object(T *Obj or T &Obj)` — specialize a
  trivially copyable object (copyable via `std::memcpy`) (all fields are treated
  as constants)

When you annotate a function or kernel with these APIs, Proteus will JIT-compile
it automatically — there is no need to also use the `annotate` attribute.

Here’s the same DAXPY example written with annotation functions:

```cpp title="Host"
void daxpy(double A, size_t N, double *X, double *Y) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
  for (std::size_t I{0}; I < N; I++) {
    Y[I] += X[I] * A;
  }
}
```

```cuda title="Device"
__global__
void daxpy(double A, size_t N, double *X, double *Y) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int stride = blockDim.x * gridDim.x;
  for(int i=tid; i<N; i+=stride)
    X[i] = A*X[i] + Y[i];
}
```

## C++ frontend API

he C++ Frontend API lets you provide source code as a string, similar to
CUDA/HIP RTC, but with a portable and higher-level interface.  It supports
runtime C++ template instantiation and string substitution for embedding runtime
values.

Here's an illustrative example to compile a DAXPY kernel through this API:

```cpp
#include <format>
#include <proteus/CppJitModule.hpp>

// Allocate and initialize input vectors A and B, and specify their size N.
double *X = ...; // Pointer to vector X
double *Y = ...; // Pointer to vector Y
double A = ...; // Scaling factor.
size_t N = ...;  // Number of elements in each vector

std::string Code = std::format(R"cpp(
  extern "C"
  __global__
  void daxpy(double *X, double *Y)
  {{
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int stride = blockDim.x * gridDim.x;
    for(int i=tid; i<{0}; i+=stride)
      X[i] = {1}*X[i] + Y[i];
  }})cpp", N, A);


CppJitModule CJM{"cuda", Code};
// Get the kernel, provide its signature as template parameters.
auto Kernel = CJM.getKernel<void(double *, double *)>("daxpy");
Kernel.launch(
  /* GridDim */ {NumBlocks, 1, 1},
  /* BlockDim */ {ThreadsPerBlock, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  X, Y);
```

We define the code in the string `Code`, which includes placeholders `{0}` and
`{1}`.
We use `std::format` (C++20) to substitute those placeholders with the runtime
values of vector size `N` and and the scaling factor `A`.
Use `extern "C"` to avoid C++ name mangling, so the kernel is retrieved by name.

We define a `CppJitModule` named `CJM` and we get the kernel handle using
`CJM.getKernel()` in `Kernel`.  Then we launch the kernel using `Kernel.launch()`
providing as arguments the grid and block dimensions, dynamic shared memory
size, and the stream, similar to the CUDA/HIP launch kernel API.

The same example for targeting the CPU, host-only execution is very similar:
instead of `getKernel()` we do `getFunction()` and execute the JIT compiled code
using `run()` instead of `launch()`.

```cpp
...
std::string Code = std::format(R"cpp(
  extern "C"
  void daxpy(double *X, double *Y)
  {
    for(int i=0; i<{0}; ++i)
      X[i] = {1}*X[i] + Y[i];
  })cpp", N, A);


CppJitModule CJM{"host", Code};
// Get the function, provide its signature as template parameters.
auto Func = CJM.getFunction<void(double *, double *)>("daxpy");
Func.run(X, Y);
```

The C++ frontend API also supports runtime instantiation of C++ templates.
Here's a simple example:

```cpp
const char *Code = R"cpp(
  template<int V>
  __global__ void foo(int *Ret) {
      *Ret = V;
  }

  template<typename T>
  __global__ void bar(T *Ret) {
      *Ret = 42;
  }

 )cpp";


int *RetFoo = ...    // Allocate in device memory.
double *RetBar = ... // Allocate in device memory.

CppJitModule CJM{"cuda", Code};
auto InstanceFoo = CJM.instantiate("foo", "3");
auto InstanceBar = CJM.instantiate("bar", "double")

InstanceFoo.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  RetFoo
);

InstanceBar.launch(
  /* GridDim */ {1, 1, 1},
  /* BlockDim */ {1, 1, 1},
  /* ShmemSize */ 0,
  /* Stream */ nullptr,
  RetBar
);
```

The `Code` block defines two kernels:

* `foo` with a non-type template parameter (int V)
* `bar` with a type template parameter (typename T)

We use `CJM.instantiate()` to create concrete instantiations by passing the
kernel name along with the template arguments as **strings**.

The returned handle can then be launched using the same `launch()` API as
before.

## DSL API
!!! warning "This is under development"
    The DSL interface is under active development and _will_ evolve to
    include additional specialization features and a more streamlined syntax.
    However, the core concepts described here are expected to remain consistent,
    even as the implementation improves.

### Introduction

Proteus offers a **DSL** API — an alternative C++ interface for generating,
managing, and executing LLVM IR modules at runtime via JIT compilation.
Supporting both CPU (host) and GPU (CUDA/HIP) targets, this API lets users to
dynamically construct code modules using a mini-language that closely resembles
natural C++ syntax.

Compared to code annotations, the DSL API offers greater flexibility,
enabling JIT-compiled code with expanded possibilities for runtime
specialization and optimization.

### Description

This interface introduces a mini-language for expressing computation, with the
following core abstractions for programmatically generating JIT code:

- **Types**: A subset of the C++ type system, including fundamental types such
    as `int`, `size_t`, `float`, `double`, and their pointer or reference forms.
- **Modules**: Analogous to translation units, modules encapsulate code and
    organize it into functions.
- **Functions**: Units of computation within modules, containing variables and
statements.
- **Variables**: Typed entities supporting operations, such as standard arithmetic
    operations (`+`, `-`, `*`, `/`) and indexed access (`[]`) for pointers.
- **Statements**: Control flow and actions within functions, including:
    - `If` (conditional execution)
    - `For` (looping)
    - `Call` / `CallBuiltin` (function or built-in calls)
    - `Ret` (function return)

These abstractions let you build JIT code at a high level — defining functions,
variables, and control flow in a style that resembles C++ while being compiled
at runtime.

### Syntax and semantics

Here's  a concise example that demonstrates how to define a JIT Module with the
DSL API, illustrating the syntax and semantics:

```cpp
auto createJitKernel(size_t N) {
  auto J = std::make_unique<JitModule>("cuda");

  // Add a kernel with the signature: void add_vectors(double *A, double *B)
  // using vector size N as a runtime constant.
  auto KernelHandle = J->addKernel<double *, double *>("add_vectors");
  auto &F = KernelHandle.F;

  // Begin the function body.
  F.beginFunction();
  {
    // Declare local variables and argument getters.
    auto &I = F.declVar<size_t>("I");
    auto &Inc = F.declVar<size_t>("Inc");
    auto [A, B] = F.getArgs();
    auto &RunConstN = F.defRuntimeConst(N);

    // Compute the global thread index.
    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);

    // Compute the stride (total number of threads).
    Inc = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);

    // Strided loop: each thread processes multiple elements.
    F.beginFor(I, I, RunConstN, Inc);
    { A[I] = A[I] + B[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}
```

This example shows how to programmatically build a CUDA kernel for vector
addition.
The API provides constructs for declaring variables, accessing
function arguments, using built-in CUDA identifiers, and defining control flow
in a familiar C++-like style.

We start by creating a JIT module `J` targeting `"cuda"`, which corresponds to an
NVIDIA GPU.
The call to `J.addKernel()` defines a kernel with the specified argument types
(using template parameters), similar to a `__global__` function in CUDA.
This returns a handle to the kernel, which is used to launch it.
The kernel function is accessed through `F`, which serves as the entry point
for code generation.

The API uses `begin*` and `end*` methods to mark code blocks.
Here, `F.beginFunction()` starts the function body, and `F.endFunction()`
finishes it.
Using braces to indicate scope is recommended for clarity.

Variables are declared using `F.declVar()`, with type as a template parameter and a
name as a string.
The vector size `N` is embedded as the runtime constant `RunConstN` defined with
`F.defRuntimeConst()`, which specializes the code to the actual value of `N` at
execution time.
Arithmetic and assignment operators work naturally on these variables.
Function arguments are accessed by `F.getArgs()`.

To implement a GPU strided loop, `I` is initialized with the global thread
index, and `Inc` with the total number of threads.
CUDA built-ins such as `threadIdx.x`, `blockIdx.x`, `blockDim.x`, and `gridDim.x` are
accessed with `F.callBuiltin`.

The loop itself is expressed with `F.beginFor()` and `F.endFor()`, operating on the
index, bounds, and stride.
Inside, we perform element-wise vector addition.
Finally, `F.ret()` emits a return.

#### JIT compilation and execution

Once the module is defined, it can be compiled and launched:

```cpp
// ... Continuing from above ...

// Allocate and initialize input vectors A and B, and specify their size N.
double *A = ...; // Pointer to vector A
double *B = ...; // Pointer to vector B
size_t N = ...;  // Number of elements in each vector

auto [J, KernelHandle] = createJitKernel(N);
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

Calling `J.compile()` finalizes the module (no further code can be added) and
builds it into an in-memory executable object.
The `KernelHandle` is then used to launch the kernel with grid and block
dimensions, plus optional launch parameters (shared memory size and stream).
Finally, `cudaDeviceSynchronize()` ensures the kernel finishes before host code
continues.

With this, you have successfully JIT-compiled and executed a kernel.
As a final note, simply changing the target from `"cuda"` to `"hip"` allows the
same code to run on an AMD GPU device.
Stay tuned for future updates, particularly around runtime specialization and
optimization!

!!! info
    This example is included in the test suite:
    - CPU: [add_vectors_runconst.cpp](https://github.com/Olympus-HPC/proteus/blob/main/tests/frontend/gpu/add_vectors_runconst.cpp)
    - GPU: [add_vectors_runconst.cpp](https://github.com/Olympus-HPC/proteus/blob/main/tests/frontend/gpu/add_vectors_runconst.cpp)
