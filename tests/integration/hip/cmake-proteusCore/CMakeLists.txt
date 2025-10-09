cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES HIP CXX C)

set(CMAKE_CXX_STANDARD 17)
find_package(hip REQUIRED CONFIG)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

set(CLANG_EXECUTABLE "${LLVM_INSTALL_DIR}/bin/clang++")
set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/test.hip")
set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/test.bc")

# CMAKE_HIP_ARCHITECTURES will contain the detected architecture(s)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "Undefined CMAKE_HIP_ARCHITECTURES")
endif()

list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
message(STATUS "HIP architecture: ${CMAKE_HIP_ARCHITECTURES}")

add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${CLANG_EXECUTABLE} -emit-llvm --offload-arch=${CMAKE_HIP_ARCHITECTURES} --offload-device-only -c ${SRC_FILE} -o ${BC_FILE}
    DEPENDS ${SRC_FILE}
    COMMENT "Generating LLVM bitcode for ${SRC_FILE}"
    VERBATIM
)

add_custom_target(generate_bc ALL DEPENDS ${BC_FILE})

add_executable(main jit.cpp)
target_link_libraries(main PRIVATE proteusCore hip::host)
target_link_options(main PRIVATE --no-undefined)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS main)
