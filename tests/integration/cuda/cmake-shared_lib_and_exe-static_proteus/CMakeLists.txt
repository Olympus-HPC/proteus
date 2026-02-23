cmake_minimum_required(VERSION 3.18)
project(libdaxpy LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(PROTEUS_ENABLE_CUDA)

# Add the shared library
add_library(daxpy SHARED libdaxpy.cpp)
target_link_libraries(daxpy PUBLIC CUDA::cudart)
set_source_files_properties(libdaxpy.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(daxpy PROPERTIES
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib")
add_proteus(daxpy)

# Make calling executable
add_executable(main main.cpp)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(main PUBLIC daxpy)

# Install command
install(TARGETS main daxpy)
