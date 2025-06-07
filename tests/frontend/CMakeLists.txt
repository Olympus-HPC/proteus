function(CREATE_FRONTEND_TEST exe check_source)
    add_executable(${exe} ${check_source} ${ARGN})

    target_link_libraries(${exe} PUBLIC proteus)

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

    add_test(NAME ${exe} COMMAND ${LIT} -vv ${check_source})
    set_property(TEST ${exe} PROPERTY LABELS "frontend")
endfunction()

function(CREATE_FRONTEND_TEST_GPU exe check_source)
    if(PROTEUS_ENABLE_HIP)
        SET(lang HIP)
    else()
        SET(lang CUDA)
    endif()

    add_executable(${exe}.${lang} ${check_source} ${ARGN})
    target_link_libraries(${exe}.${lang} PUBLIC proteus)
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE ${lang}
    )

    add_test(NAME ${exe}.${lang} COMMAND ${LIT} -vv -D EXT=${lang} -DFILECHECK=${FILECHECK} ${check_source})
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
"
)

add_subdirectory(cpu)
if(PROTEUS_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
if(PROTEUS_ENABLE_HIP)
    add_subdirectory(hip)
endif()


