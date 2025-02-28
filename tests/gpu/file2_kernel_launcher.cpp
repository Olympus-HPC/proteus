#include <stdio.h>

#include "gpu_common.h"
#include "launcher.hpp"
#include <proteus/JitInterface.hpp>

void foo() { gpuErrCheck(launcher(kernel_body)); }
