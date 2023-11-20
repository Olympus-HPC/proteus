find_program(LIT lit)

if(NOT LIT)
    message(FATAL_ERROR "Testing requires lit to be installed")
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

target_link_libraries(daxpy PUBLIC jit)
target_compile_options(
    daxpy
    PUBLIC
    -fpass-plugin=$<TARGET_FILE:JitPass>
    -march=native)

if(ENABLE_HIP)
    add_subdirectory(hip)
endif()
