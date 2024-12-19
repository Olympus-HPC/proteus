#include <stdio.h>

#include "gpu_common.h"
#include "launcher.hpp"

void foo() {
  launcher(my_kernel_body);
}
