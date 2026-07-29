import ctypes
import os

import proteus


CUDA_IR = r'''
source_filename = "gpu.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel_add(ptr %ptr, i32 %value) #0 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bdx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %sum = add i32 %tid, %bdx
  %index = sext i32 %sum to i64
  %address = getelementptr inbounds i32, ptr %ptr, i64 %index
  store i32 %value, ptr %address, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

attributes #0 = { "target-cpu"="sm_80" }

!nvvm.annotations = !{!0}
!0 = !{ptr @kernel_add, !"kernel", i32 1}
'''


HIP_IR = r'''
source_filename = "gpu.ll"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel_add(ptr addrspace(1) %ptr, i32 %value) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %bdx = call i32 @llvm.amdgcn.workgroup.size.x()
  %sum = add i32 %tid, %bdx
  %index = sext i32 %sum to i64
  %address = getelementptr inbounds i32, ptr addrspace(1) %ptr, i64 %index
  store i32 %value, ptr addrspace(1) %address, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.size.x()
'''


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _backend_settings():
    if proteus.has_cuda:
        return (
            CUDA_IR,
            os.environ.get("PROTEUS_TEST_DEVICE_ARCH", "sm_80"),
            "rtc",
        )
    if proteus.has_hip:
        return (
            HIP_IR,
            os.environ.get("PROTEUS_TEST_DEVICE_ARCH", "gfx90a"),
            "serial",
        )
    raise AssertionError("GPU staged-module test requires CUDA or HIP")


def test_argument_specialization(ir):
    module = proteus.llvm.Module.from_ir(ir)
    value = ctypes.c_int32(7)
    arguments = (ctypes.c_void_p * 2)()
    arguments[0] = ctypes.addressof(value)
    arguments[1] = ctypes.addressof(value)

    assert (
        module.specialize_arguments(
            "kernel_add", arguments=arguments, indexes=[1]
        )
        is module
    )
    assert "store i32 7" in module.to_ir()

    _assert_raises(
        ValueError,
        lambda: proteus.llvm.Module.from_ir(ir).specialize_arguments(
            "kernel_add", arguments=b"\0" * (2 * ctypes.sizeof(ctypes.c_void_p)),
            indexes=[1]
        ),
    )
    _assert_raises(
        ValueError,
        lambda: proteus.llvm.Module.from_ir(ir).specialize_arguments(
            "kernel_add",
            arguments=(ctypes.c_void_p * 1)(),
            indexes=[0],
        ),
    )
    _assert_raises(
        ValueError,
        lambda: proteus.llvm.Module.from_ir(ir).specialize_arguments(
            "kernel_add", arguments=arguments, indexes=[1, 1]
        ),
    )


def test_launch_specialization_and_codegen(ir, arch, method):
    exact = proteus.llvm.Module.from_ir(ir)
    assert (
        exact.specialize_launch_dimensions(grid=(2, 1, 1), block=(32, 1, 1))
        is exact
    )
    assert "workgroup.size.x" not in exact.to_ir().split("entry:", 1)[1].split(
        "ret void", 1
    )[0]
    assert "ntid.x" not in exact.to_ir().split("entry:", 1)[1].split(
        "ret void", 1
    )[0]

    module = proteus.llvm.Module.from_ir(ir)
    assert (
        module.assume_launch_dimension_ranges(
            grid=(2, 1, 1), block=(32, 1, 1)
        )
        is module
    )
    range_ir = module.to_ir()
    assert "range(i32 0, 32)" in range_ir or "!range" in range_ir

    codegen_module = proteus.llvm.Module.from_ir(ir)
    codegen_module.specialize_launch_dimensions(
        grid=(2, 1, 1), block=(32, 1, 1)
    )
    assert (
        codegen_module.set_launch_bounds(
            "kernel_add", max_threads_per_block=128, min_blocks_per_sm=2
        )
        is codegen_module
    )
    bounds_ir = codegen_module.to_ir()
    if proteus.has_cuda:
        assert "nvvm.maxntid" in bounds_ir or '"maxntid"' in bounds_ir
        assert "nvvm.minctasm" in bounds_ir or '"minctasm"' in bounds_ir
    else:
        assert '"amdgpu-flat-work-group-size"="1,128"' in bounds_ir
        assert '"amdgpu-waves-per-eu"="2,2"' in bounds_ir

    config = proteus.Config.current().codegen().replace(
        method=method,
        pipeline="default<O2>",
        opt_level="O2",
        codegen_opt_level=2,
    )
    assert codegen_module.optimize(arch, config=config) is codegen_module
    codegen_module.verify()
    before_codegen = codegen_module.to_bitcode()
    object_bytes = codegen_module.emit_object(arch, config=config)
    assert isinstance(object_bytes, bytes)
    assert object_bytes
    assert codegen_module.to_bitcode() == before_codegen

    if proteus.has_cuda:
        _assert_raises(
            ValueError,
            lambda: codegen_module.emit_object(
                arch, config=config.replace(method="serial")
            ),
        )


def main():
    ir, arch, method = _backend_settings()
    test_argument_specialization(ir)
    test_launch_specialization_and_codegen(ir, arch, method)
    print("test_gpu_llvm_module: ok")


if __name__ == "__main__":
    main()
