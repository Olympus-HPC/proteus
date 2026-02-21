cmake_minimum_required(VERSION 3.21)

project(mpi-shared-cache LANGUAGES CUDA CXX C)

find_package(CUDAToolkit REQUIRED)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})
find_package(MPI REQUIRED)

add_executable(mpi_shared_cache mpi_shared_cache.cpp)
set_source_files_properties(mpi_shared_cache.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(mpi_shared_cache PRIVATE MPI::MPI_CXX)
target_compile_definitions(mpi_shared_cache PRIVATE PROTEUS_ENABLE_HIP=1)

add_proteus(mpi_shared_cache)
