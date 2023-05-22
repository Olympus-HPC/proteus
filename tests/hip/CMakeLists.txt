# find_package(hip)

add_executable(
    daxpy_hip
    daxpy_hip.cpp
)

target_link_libraries(daxpy_hip PUBLIC jit)
#target_link_libraries(daxpy_hip PUBLIC hip::device)

target_compile_options(
    daxpy_hip
    PUBLIC
    -fPIC
    --offload-arch=gfx90a
    -I /opt/rocm-5.4.3/hip/include
    -L /opt/rocm-5.4.3/hip/lib/ -lamdhip64 
    -fpass-plugin=$<TARGET_FILE:JitPass>
    -x hip)

target_link_options(
    daxpy_hip
    PUBLIC
    -fPIC
    --offload-arch=gfx90a
    -L /opt/rocm-5.4.3/hip/lib/ -lamdhip64 
)