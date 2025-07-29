# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST exe check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    else()
        SET(lang CUDA)
    endif()

    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    get_target_property(PROTEUS_PASS_SOURCES ProteusPass SOURCES)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS "${PROTEUS_PASS_SOURCES}"
    )
    add_dependencies(${exe}.${lang} ProteusPass)

    target_link_libraries(${exe}.${lang} PUBLIC proteus)

    target_compile_options(
        ${exe}.${lang}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusPass>
    )

    # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe}.${lang} PROPERTIES LABELS "gpu;gpu-basic")
endfunction()

function(CREATE_GPU_TEST_RDC exe check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_executable(${exe}.${lang}.rdc ${check_source} ${ARGN})
    get_target_property(PROTEUS_PASS_SOURCES ProteusPass SOURCES)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS "${PROTEUS_PASS_SOURCES}"
    )
    add_dependencies(${exe}.${lang}.rdc ProteusPass)
    # We introduce a build dependency between the same non-RDC and RDC tests to
    # avoid collisions with parallel builds, in case the test uses the
    # project::jit_arg API. Both tests will create the same manifest JSON file
    # because its uniqueness depends on the source file, which is common for
    # both instances of the test.  The ProteusPass deletes the manifest file
    # after parsing it and if builds collide the compilation of one of the tests
    # may miss its manifest and mis-compile.
    if(TARGET ${exe}.${lang})
        add_dependencies(${exe}.${lang}.rdc ${exe}.${lang})
    endif()

    target_link_libraries(${exe}.${lang}.rdc PUBLIC proteus)

    if(PROTEUS_ENABLE_HIP)
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
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
        )

        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${exe}.${lang}.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_test(NAME ${exe}.${lang}.rdc COMMAND ${LIT} -vv -D EXT=${lang}.rdc -DFILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe}.${lang}.rdc PROPERTIES LABELS "gpu;gpu-rdc")
endfunction()

function(CREATE_GPU_TEST_RDC_LIBS exe libs check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    #message(FATAL_ERROR "libs ${libs} check_source ${check_source} ARGN ${ARGN}")

    CREATE_GPU_TEST_RDC(${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang}.rdc PRIVATE ${libs})
endfunction()

function(CREATE_GPU_TEST_LIBS exe libs check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    CREATE_GPU_TEST(${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PRIVATE ${libs})
endfunction()

# Creates a device library, which can be linked with another test. The library does not
# apply the pass
function(CREATE_GPU_LIBRARY lib source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_library(${lib} ${source})

    set_source_files_properties(${source} PROPERTIES LANGUAGE ${lang})

    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(PROTEUS_ENABLE_CUDA)
        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()
endfunction()

function(CREATE_PROTEUS_GPU_LIBRARY lib source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    elseif(PROTEUS_ENABLE_CUDA)
        SET(lang CUDA)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()

    add_library(${lib} ${source} ${ARGN})

    set_source_files_properties(${source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    get_target_property(PROTEUS_PASS_SOURCES ProteusPass SOURCES)
    set_source_files_properties(${source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS "${PROTEUS_PASS_SOURCES}"
    )
    add_dependencies(${lib} ProteusPass)
    target_link_libraries(${lib} PUBLIC proteus)

    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)
        target_compile_options(
          ${lib}
            PUBLIC
            -fgpu-rdc
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
        )

        target_link_options(${lib} PUBLIC -fgpu-rdc --hip-link)
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(
            ${lib}
            PUBLIC
            -fpass-plugin=$<TARGET_FILE:ProteusPass>
        )
        # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
    endif()
endfunction()

if(PROTEUS_ENABLE_HIP)
    enable_language(HIP)
elseif(PROTEUS_ENABLE_CUDA)
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
import os

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
ext = lit_config.params['EXT']
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%ext', ext))
config.substitutions.append(('%FILECHECK', FILECHECK))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_GPU_TEST(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST(kernel_args kernel_args.cpp)
CREATE_GPU_TEST(kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST(kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST(kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST(indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST(indirect_fallthrough indirect_fallthrough.cpp)
CREATE_GPU_TEST(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(daxpy daxpy.cpp)
CREATE_GPU_TEST(daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST(types types.cpp)
CREATE_GPU_TEST(types_api types_api.cpp)
CREATE_GPU_TEST(kernel_unused_gvar kernel_unused_gvar.cpp kernel_unused_gvar_def.cpp)
CREATE_GPU_TEST(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST(kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST(block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST(block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST(block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST(lambda lambda.cpp)
CREATE_GPU_TEST(shared_array shared_array.cpp)
CREATE_GPU_TEST(enable_disable enable_disable.cpp)
CREATE_GPU_TEST(lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST(lambda_def lambda_def.cpp)
CREATE_GPU_TEST(lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST(builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST(kernel_calls_indirect kernel_calls_indirect.cpp)
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST(alias_func alias_func.cpp)
    CREATE_GPU_TEST(alias_gvar alias_gvar.cpp)
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA support alias on func only on LLVM >=18.
    if(LLVM_VERSION_MAJOR VERSION_GREATER_EQUAL 18)
        CREATE_GPU_TEST(alias_func alias_func.cpp)
    endif()
    # CUDA does not support alias on global variables.
endif()
CREATE_GPU_TEST(mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST(types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST(dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST(jit_struct jit_struct.cpp)

CREATE_GPU_TEST_RDC(kernel kernel.cpp)
CREATE_GPU_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST_RDC(kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST_RDC(kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_arg_api indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher_multi_arg_api indirect_launcher_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST_RDC(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST_RDC(types types.cpp)
CREATE_GPU_TEST_RDC(types_api types_api.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_func kernel_calls_func.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_func_api kernel_calls_func_api.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST_RDC(kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST_RDC(kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST_RDC(kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST_RDC(multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST_RDC(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST_RDC(block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST_RDC(lambda lambda.cpp)
CREATE_GPU_TEST_RDC(shared_array shared_array.cpp)
CREATE_GPU_TEST_RDC(enable_disable enable_disable.cpp)
CREATE_GPU_TEST_RDC(lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST_RDC(lambda_def lambda_def.cpp)
CREATE_GPU_TEST_RDC(lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST_RDC(builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST_RDC(kernel_calls_indirect kernel_calls_indirect.cpp)
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST_RDC(alias_func alias_func.cpp)
    # HIP does not support alias on global variables in RDC.
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA does not support alias in RDC.
endif()

CREATE_GPU_LIBRARY(device_func_lib device_func.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib device_func_lib kernel_calls_func_lib.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib_api device_func_lib kernel_calls_func_lib_api.cpp)
CREATE_GPU_TEST_RDC(mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST_RDC(types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST_RDC(dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST_RDC(jit_struct jit_struct.cpp)

if(PROTEUS_ENABLE_HIP)
    if(LLVM_VERSION_MAJOR GREATER_EQUAL 18)
        function(CREATE_GPU_TEST_COMPRESS exe check_source)
            CREATE_GPU_TEST(${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP PRIVATE --offload-compress)
        endfunction()

        function(CREATE_GPU_TEST_RDC_COMPRESS exe check_source)
            CREATE_GPU_TEST_RDC(${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
        endfunction()

        CREATE_GPU_TEST_COMPRESS(kernel kernel.cpp)
        CREATE_GPU_TEST_RDC_COMPRESS(kernel kernel.cpp)
    endif()
endif()

add_subdirectory(scale100)
add_subdirectory(scale100-gvar)
