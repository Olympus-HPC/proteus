cmake_minimum_required(VERSION 3.18)

project(Proteus
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
option(ENABLE_HIP "Enable HIP" OFF)
option(ENABLE_CUDA "Enable CUDA" OFF)
option(ENABLE_DEBUG "Enable debugging output" OFF)
option(ENABLE_TIME_TRACING "Enable time tracing for the JIT engine" OFF)
option(BUILD_SHARED "Builds the JIT library as shared" OFF)
option(ENABLE_TESTS "Enable tests" ON)

set(LLVM_VERSION "17" CACHE STRING "LLVM version")

if(ENABLE_HIP)
  add_definitions("-DENABLE_HIP")
  find_package(hip 5.7.1 REQUIRED CONFIG)
endif()

if(ENABLE_CUDA)
  add_definitions("-DENABLE_CUDA")

  find_package(CUDAToolkit 12 REQUIRED)
endif()

if(ENABLE_DEBUG)
  add_definitions("-DENABLE_DEBUG")
endif()

include(cmake/SetupLLVM.cmake)

add_subdirectory(lib)
add_subdirectory(pass)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/proteusConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
  INSTALL_DESTINATION
  lib/cmake/proteus)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(EXPORT proteusTargets
  DESTINATION lib/cmake/proteus)

install(FILES
  "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
  DESTINATION
  lib/cmake/proteus)
