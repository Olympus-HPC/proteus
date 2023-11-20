add_executable(
    daxpy
    daxpy.cpp
)

target_link_libraries(daxpy PUBLIC jit)
target_compile_options(
    daxpy
    PUBLIC
    -fpass-plugin=$<TARGET_FILE:JitPass>
    -march=native)

if(ENABLE_HIP)
    add_subdirectory(hip)
endif()
