target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @write42(ptr addrspace(1) %out) {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @write42, !"kernel", i32 1}
