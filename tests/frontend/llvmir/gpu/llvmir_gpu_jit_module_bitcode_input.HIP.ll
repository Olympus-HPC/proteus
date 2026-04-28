target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @write42(ptr addrspace(1) %out) #0 {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
