cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES CUDA CXX C)

find_package(CUDAToolkit REQUIRED)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

# libfoo
set_source_files_properties(libfoo.cpp PROPERTIES LANGUAGE CUDA)
add_library(libfoo SHARED libfoo.cpp)
target_compile_options(libfoo PRIVATE --no-offload-new-driver)
set_target_properties(libfoo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libfoo PROPERTIES
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib")
add_proteus(libfoo)

# libbar
set_source_files_properties(libbar.cpp PROPERTIES LANGUAGE CUDA)
add_library(libbar SHARED libbar.cpp)
target_compile_options(libbar PRIVATE --no-offload-new-driver)
set_target_properties(libbar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libbar PROPERTIES
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib")
add_proteus(libbar)

# main
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(main main.cpp)
target_compile_options(main PRIVATE --no-offload-new-driver)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(main PUBLIC libfoo libbar)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
add_proteus(main)

install(TARGETS main libfoo libbar)
