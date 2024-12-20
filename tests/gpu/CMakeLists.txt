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
        OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/pass/ProteusPass.cpp
    )
    add_dependencies(${exe}.${lang} ProteusPass)

    target_link_libraries(${exe}.${lang} PUBLIC proteus)

    target_compile_options(
        ${exe}.${lang}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusPass>
    )

    # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} ${check_source})
    set_tests_properties(${exe}.${lang} PROPERTIES FIXTURES_REQUIRED setupDeleteJitStoredCacheFiles)
    set_tests_properties(${exe}.${lang} PROPERTIES FIXTURES_REQUIRED cleanupDeleteJitStoredCacheFiles)
endfunction()

function(CREATE_GPU_TEST_RDC exe check_source)
    if(ENABLE_HIP)
        SET(lang HIP)
    elseif(ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_executable(${exe}.${lang}.rdc ${check_source} ${ARGN})
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/pass/ProteusPass.cpp
    )

    target_link_libraries(${exe}.${lang}.rdc PUBLIC proteus)

    if(ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

        # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
        )

        target_link_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            --hip-link
            -Xoffload-linker --load-pass-plugin=$<TARGET_FILE:ProteusPass>
        )
    elseif(ENABLE_CUDA)
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
        )

        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${exe}.${lang}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_test(NAME ${exe}.${lang}.rdc COMMAND ${LIT} -vv -D EXT=${lang}.rdc ${check_source})
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES FIXTURES_REQUIRED setupDeleteJitStoredCacheFiles)
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES FIXTURES_REQUIRED cleanupDeleteJitStoredCacheFiles)
endfunction()

function(CREATE_GPU_TEST_RDC_LIBS exe libs check_source)
    if(ENABLE_HIP)
        SET(lang HIP)
    elseif(ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    #message(FATAL_ERROR "libs ${libs} check_source ${check_source} ARGN ${ARGN}")

    CREATE_GPU_TEST_RDC(${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang}.rdc PRIVATE ${libs})
endfunction()

# Creates a device library, which can be linked with another test. The library does not
# apply the pass
function(CREATE_GPU_LIBRARY lib source)
    if(ENABLE_HIP)
        SET(lang HIP)
    elseif(ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_library(${lib} ${source})

    set_source_files_properties(${source} PROPERTIES LANGUAGE ${lang})

    if(ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(ENABLE_CUDA)
        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "ENABLE_HIP or ENABLE_CUDA must be defined for GPU tests")
    endif()
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
CREATE_GPU_TEST(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST(indirect_fallthrough indirect_fallthrough.cpp)
CREATE_GPU_TEST(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(daxpy daxpy.cpp)
CREATE_GPU_TEST(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST(types types.cpp)
CREATE_GPU_TEST(kernel_unused_gvar kernel_unused_gvar.cpp kernel_unused_gvar_def.cpp)
CREATE_GPU_TEST(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)

CREATE_GPU_TEST_RDC(kernel kernel.cpp)
CREATE_GPU_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST_RDC(daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST_RDC(types types.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_func kernel_calls_func.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST_RDC(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST_RDC(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST_RDC(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)

CREATE_GPU_LIBRARY(device_func_lib device_func.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib device_func_lib kernel_calls_func_lib.cpp)
