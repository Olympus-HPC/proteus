if(PROTEUS_ENABLE_HIP)
    SET(lang HIP)
elseif(PROTEUS_ENABLE_CUDA)
    SET(lang CUDA)
else()
    message(FATAL_ERROR "PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA must be defined for GPU tests")
endif()

# The first argument is the executable to generate, the second arguments is the
# source file with checks. Any following arguments are other source files
# (without checks) to compile for generating the executable.
function(CREATE_GPU_TEST ENABLE_AGGRESSIVE exe check_source)
    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    get_target_property(PROTEUS_PASS_SOURCES ProteusPass SOURCES)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS "${PROTEUS_PASS_SOURCES}"
    )
    add_dependencies(${exe}.${lang} ProteusPass)

    target_link_libraries(${exe}.${lang} PUBLIC proteus)
    if("${ENABLE_AGGRESSIVE}" STREQUAL "OFF")
       add_proteus(${exe}.${lang})
    else()
    	add_proteus(${exe}.${lang} ENABLE_AGGRESSIVE)
    endif()

    # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe}.${lang} PROPERTIES LABELS "gpu;gpu-basic")
endfunction()

function(CREATE_GPU_TEST_RDC ENABLE_AGGRESSIVE exe check_source)
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
    if("${ENABLE_AGGRESSIVE}" STREQUAL "OFF")
       add_proteus(${exe}.${lang}.rdc)
    else()
    	add_proteus(${exe}.${lang}.rdc ENABLE_AGGRESSIVE)
    endif()


    if(PROTEUS_ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

        # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
        )

        target_link_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            --hip-link
        )
    elseif(PROTEUS_ENABLE_CUDA)
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
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
    #message(FATAL_ERROR "libs ${libs} check_source ${check_source} ARGN ${ARGN}")

    CREATE_GPU_TEST_RDC(OFF ${exe} ${check_source} ${ARGN})
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

    CREATE_GPU_TEST(OFF ${exe} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PRIVATE ${libs})
endfunction()

# Creates a device library, which can be linked with another test. The library does not
# apply the pass
function(CREATE_GPU_LIBRARY lib source)
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
    add_library(${lib} ${source} ${ARGN})

    set_source_files_properties(${source} ${ARGN} PROPERTIES LANGUAGE ${lang})
    get_target_property(PROTEUS_PASS_SOURCES ProteusPass SOURCES)
    set_source_files_properties(${source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
        OBJECT_DEPENDS "${PROTEUS_PASS_SOURCES}"
    )
    add_dependencies(${lib} ProteusPass)
    target_link_libraries(${lib} PUBLIC proteus)
    add_proteus(${lib})
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
import tempfile
import atexit
import shutil

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
# Create a unique temp exec_root to avoid races on lit_test_times.txt
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

ext = lit_config.params['EXT']
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%ext', ext))
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
config.substitutions.append(('%device_lang', '${lang}'))
"
)

CREATE_GPU_TEST(OFF kernel kernel.cpp)
CREATE_GPU_TEST(OFF kernel_pass_pipeline kernel_pass_pipeline.cpp)
CREATE_GPU_TEST(OFF kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST(OFF kernel_args kernel_args.cpp)
CREATE_GPU_TEST(OFF kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST(OFF kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST(OFF kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST(OFF kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST(OFF kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST(OFF kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST(OFF indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST(OFF indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST(OFF indirect_fallthrough indirect_fallthrough.cpp)
CREATE_GPU_TEST(OFF inlining_regression inlining_regression.cpp)
CREATE_GPU_TEST(OFF multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(OFF daxpy daxpy.cpp)
CREATE_GPU_TEST(OFF daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST(OFF kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(OFF kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST(OFF kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST(OFF types types.cpp)
CREATE_GPU_TEST(OFF types_api types_api.cpp)
CREATE_GPU_TEST(OFF kernel_unused_gvar kernel_unused_gvar.cpp kernel_unused_gvar_def.cpp)
CREATE_GPU_TEST(OFF kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST(OFF kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST(OFF kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST(OFF kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST(OFF multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST(OFF block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST(OFF block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST(OFF block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST(OFF lambda lambda.cpp)
CREATE_GPU_TEST(OFF shared_array shared_array.cpp)
CREATE_GPU_TEST(OFF enable_disable enable_disable.cpp)
CREATE_GPU_TEST(OFF lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST(OFF lambda_def lambda_def.cpp)
CREATE_GPU_TEST(OFF lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST(OFF builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST(OFF kernel_calls_indirect kernel_calls_indirect.cpp)
CREATE_GPU_TEST(ON aggressive_annotations proteus_aggressive_annot.cpp)
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST(OFF alias_func alias_func.cpp)
    CREATE_GPU_TEST(OFF alias_gvar alias_gvar.cpp)
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA support alias on func only on LLVM >=18.
    if(LLVM_VERSION_MAJOR VERSION_GREATER_EQUAL 18)
       CREATE_GPU_TEST(OFF alias_func alias_func.cpp)
    endif()
    # CUDA does not support alias on global variables.
endif()
CREATE_GPU_TEST(OFF mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST(OFF types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST(OFF dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST(OFF jit_struct jit_struct.cpp)
CREATE_GPU_TEST(OFF kernel_tuning kernel_tuning.cpp)

CREATE_GPU_TEST_RDC(OFF kernel kernel.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_pass_pipeline kernel_pass_pipeline.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_args_api kernel_args_api.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_args_annot_long kernel_args_annot_long.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_args_annot_mix kernel_args_annot_mix.cpp)
CREATE_GPU_TEST_RDC(OFF kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_arg indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_arg_api indirect_launcher_arg.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_tpl_multi indirect_launcher_tpl_multi.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_tpl_multi_arg indirect_launcher_tpl_multi_arg.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_tpl_multi_arg_api indirect_launcher_tpl_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_multi indirect_launcher_multi.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_multi_arg indirect_launcher_multi_arg.cpp)
CREATE_GPU_TEST_RDC(OFF indirect_launcher_multi_arg_api indirect_launcher_multi_arg_api.cpp)
CREATE_GPU_TEST_RDC(OFF daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(OFF daxpy_api daxpy_api.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_host_device_jit kernel_host_device_jit.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_host_device_jit_api kernel_host_device_jit_api.cpp)
CREATE_GPU_TEST_RDC(OFF types types.cpp)
CREATE_GPU_TEST_RDC(OFF types_api types_api.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_calls_func kernel_calls_func.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_calls_func_api kernel_calls_func_api.cpp device_func.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_repeat kernel_repeat.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_repeat_api kernel_repeat_api.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_launch_exception kernel_launch_exception.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_preset_bounds kernel_preset_bounds.cpp)
CREATE_GPU_TEST_RDC(OFF multi_file_launcher file1_kernel_launcher.cpp file2_kernel_launcher.cpp)
CREATE_GPU_TEST_RDC(OFF multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST_RDC(OFF block_grid_dim_1d block_grid_dim_1d.cpp)
CREATE_GPU_TEST_RDC(OFF block_grid_dim_2d block_grid_dim_2d.cpp)
CREATE_GPU_TEST_RDC(OFF block_grid_dim_3d block_grid_dim_3d.cpp)
CREATE_GPU_TEST_RDC(OFF lambda lambda.cpp)
CREATE_GPU_TEST_RDC(OFF shared_array shared_array.cpp)
CREATE_GPU_TEST_RDC(OFF enable_disable enable_disable.cpp)
CREATE_GPU_TEST_RDC(OFF lambda_multiple lambda_multiple.cpp)
CREATE_GPU_TEST_RDC(OFF lambda_def lambda_def.cpp)
CREATE_GPU_TEST_RDC(OFF lambda_host_device lambda_host_device.cpp)
CREATE_GPU_TEST_RDC(OFF builtin_globals builtin_globals.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_calls_indirect kernel_calls_indirect.cpp)
CREATE_GPU_TEST_RDC(ON aggressive_annotations proteus_aggressive_annot.cpp)
if(PROTEUS_ENABLE_HIP)
    CREATE_GPU_TEST_RDC(OFF alias_func alias_func.cpp)
    # HIP does not support alias on global variables in RDC.
elseif(PROTEUS_ENABLE_CUDA)
    # CUDA does not support alias in RDC.
endif()

CREATE_GPU_LIBRARY(device_func_lib device_func.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib device_func_lib kernel_calls_func_lib.cpp)
CREATE_GPU_TEST_RDC_LIBS(kernel_calls_func_lib_api device_func_lib kernel_calls_func_lib_api.cpp)
CREATE_GPU_TEST_RDC(OFF mix_attr_api mix_attr_api.cpp)
CREATE_GPU_TEST_RDC(OFF types_jit_array types_jit_array.cpp)
CREATE_GPU_TEST_RDC(OFF dynamic_jit_array dynamic_jit_array.cpp)
CREATE_GPU_TEST_RDC(OFF jit_struct jit_struct.cpp)
CREATE_GPU_TEST_RDC(OFF kernel_tuning kernel_tuning.cpp)

if(PROTEUS_ENABLE_HIP)
    if(LLVM_VERSION_MAJOR GREATER_EQUAL 18)
        function(CREATE_GPU_TEST_COMPRESS exe check_source)
	    CREATE_GPU_TEST(OFF ${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP PRIVATE --offload-compress)
        endfunction()

        function(CREATE_GPU_TEST_RDC_COMPRESS exe check_source)
            CREATE_GPU_TEST_RDC(OFF ${exe}.compress ${check_source} ${ARGN})
            target_compile_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
            target_link_options(${exe}.compress.HIP.rdc PRIVATE --offload-compress)
        endfunction()

        CREATE_GPU_TEST_COMPRESS(kernel kernel.cpp)
        CREATE_GPU_TEST_RDC_COMPRESS(kernel kernel.cpp)
    endif()
endif()

add_subdirectory(scale100)
add_subdirectory(scale100-gvar)
