import os
import subprocess
import sys

import proteus


def expect_raises(exc_type, func, match=None):
    try:
        func()
    except exc_type as exc:
        if match is not None:
            assert match in str(exc), str(exc)
        return exc
    else:
        raise AssertionError(f"expected {exc_type.__name__}")


def expect_subprocess_failure(code):
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)
    return result


def get_gpu_target():
    if proteus.has_cuda:
        return "cuda"
    if proteus.has_hip:
        return "hip"
    return None


def get_cpp_gpu_source():
    target = get_gpu_target()
    if target == "cuda":
        include = "#include <cuda_runtime.h>"
    elif target == "hip":
        include = "#include <hip/hip_runtime.h>"
    else:
        return None, None

    source = (
        include
        + r'''
extern "C" __global__ void write_int(int *out, int value) {
  *out = value;
}
'''
    )
    return target, source


def get_mlir_gpu_source():
    return r'''
module attributes {gpu.container_module} {
  gpu.module @user_device_module {
    gpu.func @write42(%out: !llvm.ptr) kernel {
      %c42 = arith.constant 42 : i32
      llvm.store %c42, %out : i32, !llvm.ptr
      gpu.return
    }
  }
}
'''


class DataPtr:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


class CudaArrayInterface:
    def __init__(self, ptr):
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "<i4",
            "data": (ptr, False),
            "version": 3,
        }


class EmptyCudaArrayInterface:
    __cuda_array_interface__ = {
        "shape": (1,),
        "typestr": "<i4",
        "data": (),
        "version": 3,
    }
