# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
    if(ENABLE_HIP)
        SET(lang HIP)
    else()
        SET(lang CUDA)
    endif()

    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/ProteusJitPass.cpp
    )
    add_dependencies(${exe}.${lang} ProteusJitPass)

    target_link_libraries(${exe}.${lang} PUBLIC proteusjit)

    target_compile_options(
        ${exe}.${lang}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusJitPass>
    )

    # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} ${check_source})
    set_tests_properties(${exe}.${lang} PROPERTIES FIXTURES_REQUIRED setupDeleteJitStoredCacheFiles)
    set_tests_properties(${exe}.${lang} PROPERTIES FIXTURES_REQUIRED cleanupDeleteJitStoredCacheFiles)
endfunction()

function(CREATE_GPU_TEST_RDC exe source)
    if(ENABLE_HIP)
        SET(lang HIP)
    elseif(ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_executable(${exe}.${lang}.rdc ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/ProteusJitPass.cpp
    )
    add_dependencies(${exe}.${lang}.rdc ProteusJitPass)

    target_link_libraries(${exe}.${lang}.rdc PUBLIC proteusjit)

    if(ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

        # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            -fpass-plugin=$<TARGET_FILE:ProteusJitPass>
        )

        target_link_options(${exe}.${lang}.rdc PUBLIC -fgpu-rdc --hip-link)
    elseif(ENABLE_CUDA)
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fpass-plugin=$<TARGET_FILE:ProteusJitPass>
        )

        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${exe}.${lang}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_test(NAME ${exe}.${lang}.rdc COMMAND ${LIT} -vv -D EXT=${lang}.rdc ${source})
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES FIXTURES_REQUIRED setupDeleteJitStoredCacheFiles)
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES FIXTURES_REQUIRED cleanupDeleteJitStoredCacheFiles)
endfunction()

if(ENABLE_HIP)
    enable_language(HIP)
elseif(ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
    endif()

    enable_language(CUDA)
    message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

    if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
        message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
    endif()
endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
ext = lit_config.params['EXT']
config.substitutions.append(('%ext', ext))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_GPU_TEST(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST(kernel_args kernel_args.cpp)
CREATE_GPU_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST(void_indirect_launcher void_indirect_launcher.cpp)
CREATE_GPU_TEST(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(daxpy daxpy.cpp)
CREATE_GPU_TEST(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST(types types.cpp)
CREATE_GPU_TEST(kernel_unused_gvar kernel_unused_gvar.cpp kernel_unused_gvar_def.cpp)
CREATE_GPU_TEST(launch_in_if_else launch_in_if_else.cpp)

CREATE_GPU_TEST_RDC(kernel kernel.cpp)
CREATE_GPU_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST_RDC(types types.cpp)

# TODO: Those two tests fail on CUDA with JIT stored caching because device
# global variables need to be re-linked to their runtime address. However, I
# chose NOT to cast them as WILL_FAIL with the code here, to make the failure
# explicit when running tests. The WIP solution is caching LLVM IR instead of a
# binary object, which is pending implementation.
#if(ENABLE_CUDA)
#    set_tests_properties(kernels_gvar.CUDA kernels_gvar.CUDA.rdc PROPERTIES WILL_FAIL TRUE)
#endif()
