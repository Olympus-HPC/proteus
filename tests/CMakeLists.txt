find_program(LIT lit
    PATHS ${LLVM_INSTALL_DIR}/bin)

if(NOT LIT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import lit"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    if(${EXIT_CODE} EQUAL 0)
        set(LIT ${CMAKE_CURRENT_SOURCE_DIR}/lit-main.py)
        message(STATUS "Using ${LIT} for lit testing")
    else()
        message(FATAL_ERROR "Testing requires lit to be installed, e.g., python -m pip install lit>=15.0.7")
    endif()
else()
    message(STATUS "Found lit at ${LIT}")
endif()

find_program(FILECHECK FileCheck
    PATHS ${LLVM_INSTALL_DIR}/bin)

if(NOT FILECHECK)
    message(FATAL_ERROR "Testing requires FileCheck to be installed")
else()
    message(STATUS "Found Filecheck at ${FILECHECK}")
endif()

add_executable(
    daxpy
    daxpy.cpp
)

set_source_files_properties(daxpy.cpp PROPERTIES
    OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
)
add_dependencies(daxpy JitPass)

target_link_libraries(daxpy PUBLIC jit)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
    set(NATIVE_OPT_FLAGS "-mcpu=native" "-mtune=native")
else()
    set(NATIVE_OPT_FLAGS "-march=native")
endif()

target_compile_options(
    daxpy
    PUBLIC
    -fpass-plugin=$<TARGET_FILE:JitPass>
    ${NATIVE_OPT_FLAGS}
)

add_test(NAME setupDeleteJitStoredCache COMMAND sh -c "rm -f ${CMAKE_CURRENT_BINARY_DIR}/gpu/cache-jit-*.o")
add_test(NAME cleanupDeleteJitStoredCache COMMAND sh -c "rm -f ${CMAKE_CURRENT_BINARY_DIR}/gpu/cache-jit-*.o")
set_tests_properties(setupDeleteJitStoredCache PROPERTIES FIXTURES_SETUP setupDeleteJitStoredCacheFiles)
set_tests_properties(cleanupDeleteJitStoredCache PROPERTIES FIXTURES_CLEANUP cleanupDeleteJitStoredCacheFiles)

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
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )
    add_dependencies(${exe}.${lang} JitPass)

    target_link_libraries(${exe}.${lang} PUBLIC jit)

    target_compile_options(
        ${exe}.${lang}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:JitPass>
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
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/JitPass.cpp
    )
    add_dependencies(${exe}.${lang}.rdc JitPass)

    target_link_libraries(${exe}.${lang}.rdc PUBLIC jit)

    if(ENABLE_HIP)
        # This is unsupported see: https://gitlab.kitware.com/cmake/cmake/-/issues/23210
        # set_target_properties(${exe}.rdc PROPERTIES HIP_SEPARABLE_COMPILATION on)

        # Add -fgpu-rdc, --hip-link options for compilation and linking to enable RDC.
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fgpu-rdc
            -fpass-plugin=$<TARGET_FILE:JitPass>
        )

        target_link_options(${exe}.${lang}.rdc PUBLIC -fgpu-rdc --hip-link)
    elseif(ENABLE_CUDA)
        target_compile_options(
            ${exe}.${lang}.rdc
            PUBLIC
            -fpass-plugin=$<TARGET_FILE:JitPass>
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

if(ENABLE_HIP OR ENABLE_CUDA)
    add_subdirectory(gpu)
endif()
