import ctypes
import ctypes.util
import unittest

import proteus


class CtypesRuntime:
    success = 0

    def __init__(self, libname, prefix, h2d_kind, d2h_kind):
        self.lib = ctypes.CDLL(libname)
        self.malloc = getattr(self.lib, f"{prefix}Malloc")
        self.memcpy = getattr(self.lib, f"{prefix}Memcpy")
        self.sync_call = getattr(self.lib, f"{prefix}DeviceSynchronize")
        self.free_call = getattr(self.lib, f"{prefix}Free")
        self.h2d_kind = h2d_kind
        self.d2h_kind = d2h_kind

        self.malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.malloc.restype = ctypes.c_int
        self.memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.memcpy.restype = ctypes.c_int
        self.sync_call.argtypes = []
        self.sync_call.restype = ctypes.c_int
        self.free_call.argtypes = [ctypes.c_void_p]
        self.free_call.restype = ctypes.c_int

    def check(self, err):
        if err != self.success:
            raise RuntimeError(f"{self.name} runtime call failed with error {err}")

    def malloc_i32(self):
        ptr = ctypes.c_void_p()
        self.check(self.malloc(ctypes.byref(ptr), ctypes.sizeof(ctypes.c_int)))
        return ptr.value

    def copy_h2d_i32(self, ptr, value):
        host = ctypes.c_int(value)
        self.check(
            self.memcpy(
                ctypes.c_void_p(ptr),
                ctypes.byref(host),
                ctypes.sizeof(host),
                self.h2d_kind,
            )
        )

    def copy_d2h_i32(self, ptr):
        host = ctypes.c_int()
        self.check(
            self.memcpy(
                ctypes.byref(host),
                ctypes.c_void_p(ptr),
                ctypes.sizeof(host),
                self.d2h_kind,
            )
        )
        return host.value

    def sync(self):
        self.check(self.sync_call())

    def free(self, ptr):
        self.check(self.free_call(ctypes.c_void_p(ptr)))


class CudaCtypesRuntime(CtypesRuntime):
    name = "CUDA"

    def __init__(self):
        libname = ctypes.util.find_library("cudart") or "libcudart.so"
        super().__init__(libname, "cuda", h2d_kind=1, d2h_kind=2)


class HipCtypesRuntime(CtypesRuntime):
    name = "HIP"

    def __init__(self):
        libname = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
        super().__init__(libname, "hip", h2d_kind=1, d2h_kind=2)


class CudaPythonRuntime:
    def __init__(self):
        from cuda import cudart
        self.cudart = cudart

    def check(self, err):
        if isinstance(err, tuple):
            err = err[0]
        if int(err) != int(self.cudart.cudaError_t.cudaSuccess):
            raise RuntimeError(f"CUDA runtime call failed with error {err}")

    def malloc_i32(self):
        err, ptr = self.cudart.cudaMalloc(ctypes.sizeof(ctypes.c_int))
        self.check(err)
        return int(ptr)

    def copy_h2d_i32(self, ptr, value):
        host = ctypes.c_int(value)
        err = self.cudart.cudaMemcpy(
            ptr,
            ctypes.addressof(host),
            ctypes.sizeof(host),
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        self.check(err)

    def copy_d2h_i32(self, ptr):
        host = ctypes.c_int()
        err = self.cudart.cudaMemcpy(
            ctypes.addressof(host),
            ptr,
            ctypes.sizeof(host),
            self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        self.check(err)
        return host.value

    def sync(self):
        self.check(self.cudart.cudaDeviceSynchronize())

    def free(self, ptr):
        self.check(self.cudart.cudaFree(ptr))


class HipPythonRuntime:
    def __init__(self):
        from hip import hip

        self.hip = hip
        self.allocations = {}

    def check(self, call_result):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
        if isinstance(err, self.hip.hipError_t):
            ok = err == self.hip.hipError_t.hipSuccess
        else:
            ok = int(err) == 0
        if not ok:
            raise RuntimeError(f"HIP runtime call failed with error {err}")
        return result

    def ptr_value(self, obj):
        if isinstance(obj, int):
            return obj
        if hasattr(obj, "data_ptr"):
            return int(obj.data_ptr())
        if hasattr(obj, "__cuda_array_interface__"):
            return int(obj.__cuda_array_interface__["data"][0])
        return int(obj)

    def malloc_i32(self):
        device = self.check(self.hip.hipMalloc(ctypes.sizeof(ctypes.c_int)))
        ptr = self.ptr_value(device)
        self.allocations[ptr] = device
        return ptr

    def device_arg(self, ptr):
        return self.allocations.get(ptr, ptr)

    def copy_h2d_i32(self, ptr, value):
        host = ctypes.c_int(value)
        self.check(
            self.hip.hipMemcpy(
                self.device_arg(ptr),
                host,
                ctypes.sizeof(host),
                self.hip.hipMemcpyKind.hipMemcpyHostToDevice,
            )
        )

    def copy_d2h_i32(self, ptr):
        host = ctypes.c_int()
        self.check(
            self.hip.hipMemcpy(
                host,
                self.device_arg(ptr),
                ctypes.sizeof(host),
                self.hip.hipMemcpyKind.hipMemcpyDeviceToHost,
            )
        )
        return host.value

    def sync(self):
        self.check(self.hip.hipDeviceSynchronize())

    def free(self, ptr):
        device = self.allocations.pop(ptr, ptr)
        self.check(self.hip.hipFree(device))


def create_runtime():
    if proteus.has_cuda:
        try:
            return CudaPythonRuntime()
        except ImportError:
            return CudaCtypesRuntime()
    if proteus.has_hip:
        try:
            return HipPythonRuntime()
        except ImportError:
            return HipCtypesRuntime()
    raise unittest.SkipTest("GPU runtime tests require CUDA or HIP")
