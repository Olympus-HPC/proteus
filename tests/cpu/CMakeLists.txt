function(CREATE_CPU_TEST exe check_source)
    add_executable(${exe} ${check_source} ${ARGN})

    target_link_libraries(${exe} PUBLIC proteus)
    proteus_attach_pass_plugin_rebuild_dep(${exe} ${check_source})

    if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
        set(NATIVE_OPT_FLAGS "-mcpu=native" "-mtune=native")
    else()
        set(NATIVE_OPT_FLAGS "-march=native")
    endif()

    target_compile_options(
        ${exe}
        PUBLIC
        ${NATIVE_OPT_FLAGS}
    )

    target_link_options(${exe} PRIVATE
        $<LINK_ONLY:-rdynamic>
    )

    add_test(NAME ${exe} COMMAND ${LIT} -vv -D FILECHECK=${FILECHECK} ${check_source})
    set_tests_properties(${exe} PROPERTIES LABELS "cpu")
endfunction()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os
import platform
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

FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%target_arch', platform.machine()))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
"
)

CREATE_CPU_TEST(daxpy daxpy.cpp)
CREATE_CPU_TEST(daxpy_api daxpy_api.cpp)
CREATE_CPU_TEST(daxpy_annot_long daxpy_annot_long.cpp)
CREATE_CPU_TEST(daxpy_annot_mix daxpy_annot_mix.cpp)
CREATE_CPU_TEST(types types.cpp)
CREATE_CPU_TEST(types_api types_api.cpp)
CREATE_CPU_TEST(lambda lambda.cpp)
CREATE_CPU_TEST(lambda_def lambda_def.cpp)
CREATE_CPU_TEST(lambda_multiple lambda_multiple.cpp)
CREATE_CPU_TEST(lambda_multiple_api lambda_multiple_api.cpp)
CREATE_CPU_TEST(lambda_spec_test lambda_spec_test.cpp)
CREATE_CPU_TEST(types_jit_array types_jit_array.cpp)
CREATE_CPU_TEST(dynamic_jit_array dynamic_jit_array.cpp)
CREATE_CPU_TEST(jit_struct jit_struct.cpp)
CREATE_CPU_TEST(custom_pipeline custom_pipeline.cpp)
CREATE_CPU_TEST(modify_gvar modify_gvar.cpp)
CREATE_CPU_TEST(jit_eh jit_eh.cpp)
