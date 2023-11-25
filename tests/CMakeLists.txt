find_program(LIT lit)

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
        message(FATAL_ERROR "Exit code ${EXIT_CODE} Testing requires lit to be installed")
    endif()
else()
    message(STATUS "Found lit at ${LIT}")
endif()

find_program(FILECHECK FileCheck)

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

if(ENABLE_HIP)
    add_subdirectory(hip)
endif()

if(ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
