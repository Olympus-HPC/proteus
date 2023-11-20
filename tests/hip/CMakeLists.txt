enable_language(HIP)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
"
)

function(CREATE_TEST exe source)
    add_executable(${exe} ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE HIP
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )

    target_link_libraries(${exe} PUBLIC jit)

    target_compile_options(
        ${exe}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )

    add_test(NAME ${exe} COMMAND ${LIT} ${source})
endfunction()

function(CREATE_TEST_RDC exe source)
    add_executable(${exe}.rdc ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE HIP
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )

    target_link_libraries(${exe}.rdc PUBLIC jit)

    # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
    # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

    # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
    target_compile_options(
        ${exe}.rdc
        PUBLIC
        -fgpu-rdc
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )

    target_link_options(${exe}.rdc PUBLIC -fgpu-rdc --hip-link)

    add_test(NAME ${exe}.rdc COMMAND ${LIT} ${source})
endfunction()

CREATE_TEST(kernel kernel.cpp)
CREATE_TEST(kernel_cache kernel_cache.cpp)
CREATE_TEST(kernel_args kernel_args.cpp)
CREATE_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_TEST(daxpy_hip daxpy_hip.cpp)

CREATE_TEST_RDC(kernel kernel.cpp)
CREATE_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_TEST_RDC(daxpy_hip daxpy_hip.cpp)
