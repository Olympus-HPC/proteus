find_package(hip)

add_executable(
    daxpy_hip
    daxpy_hip.cpp
)

target_link_libraries(daxpy_hip PUBLIC jit)
target_link_libraries(daxpy_hip PUBLIC hip::device)

target_compile_options(
    daxpy_hip
    PUBLIC
    -fpass-plugin=$<TARGET_FILE:JitPass>)