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

function(CREATE_FRONTEND_TEST_GPU exe check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    else()
        SET(lang CUDA)
    endif()

    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PUBLIC proteusFrontend)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
    )

    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
    set_property(TEST ${exe}.${lang} PROPERTY LABELS "frontend-gpu")
endfunction()

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

CREATE_FRONTEND_TEST_GPU(operators operators.cpp)
CREATE_FRONTEND_TEST_GPU(if if.cpp)
CREATE_FRONTEND_TEST_GPU(for for.cpp)
CREATE_FRONTEND_TEST_GPU(add_vectors add_vectors.cpp)
CREATE_FRONTEND_TEST_GPU(add_vectors_runconst add_vectors_runconst.cpp)
CREATE_FRONTEND_TEST_GPU(adam adam.cpp)
CREATE_FRONTEND_TEST_GPU(adam_runconst adam_runconst.cpp)
CREATE_FRONTEND_TEST_GPU(kernel_2d kernel_2d.cpp)
CREATE_FRONTEND_TEST_GPU(kernel_3d kernel_3d.cpp)
CREATE_FRONTEND_TEST_GPU(floydwarshall floydwarshall.cpp)
CREATE_FRONTEND_TEST_GPU(tiled_matmul tiled_matmul.cpp)