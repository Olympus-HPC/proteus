if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
endif()

enable_language(CUDA)

message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
    message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
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

function(CREATE_TEST exe source)
    add_executable(${exe}.cuda ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE CUDA
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )

    target_link_libraries(${exe}.cuda PUBLIC jit)

    target_compile_options(
        ${exe}.cuda
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )

    add_test(NAME ${exe}.cuda COMMAND ${LIT} -vv -D EXT=cuda ${source})
endfunction()

function(CREATE_TEST_RDC exe source)
    add_executable(${exe}.cuda.rdc ${source})
    set_source_files_properties(${source} PROPERTIES
        LANGUAGE CUDA
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )

    target_link_libraries(${exe}.cuda.rdc PUBLIC jit)

    target_compile_options(
        ${exe}.cuda.rdc
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:JitPass>
    )
    set_target_properties(${exe}.cuda.rdc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_test(NAME ${exe}.cuda.rdc COMMAND ${LIT} -vv -D EXT=cuda.rdc ${source})
endfunction()

CREATE_TEST(kernel kernel.cpp)
CREATE_TEST(kernel_cache kernel_cache.cpp)
CREATE_TEST(kernel_args kernel_args.cpp)
CREATE_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_TEST(daxpy daxpy.cpp)

CREATE_TEST_RDC(kernel kernel.cpp)
CREATE_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_TEST_RDC(daxpy daxpy.cpp)
