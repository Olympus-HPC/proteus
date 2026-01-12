#include <stdio.h>

#include "gpu_common.h"
#include "launcher.h"
#include <proteus/JitInterface.h>

void foo() { gpuErrCheck(launcher(kernel_body)); }
