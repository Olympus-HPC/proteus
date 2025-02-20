#include <hip/hip_runtime.h>

__global__ void foo(int a){
  printf("Hi from %d", a);
}
