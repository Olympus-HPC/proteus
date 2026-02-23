cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE proteusFrontend CUDA::cudart)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS main)
