enable_language(HIP)

add_executable(
    daxpy_hip
    daxpy_hip.cpp
)
set_source_files_properties(daxpy_hip.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(daxpy_hip PUBLIC jit)

target_compile_options(
    daxpy_hip
    PUBLIC
    -fpass-plugin=$<TARGET_FILE:JitPass>
)