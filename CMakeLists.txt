cmake_minimum_required(VERSION 3.18)

project(Proteus
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
option(PROTEUS_ENABLE_HIP "Enable HIP" OFF)
option(PROTEUS_ENABLE_CUDA "Enable CUDA" OFF)
option(PROTEUS_ENABLE_DEBUG "Enable debugging output" OFF)
option(PROTEUS_ENABLE_TIME_TRACING "Enable time tracing for the JIT engine" OFF)
option(BUILD_SHARED "Builds the JIT library as shared" OFF)
option(ENABLE_TESTS "Enable tests" ON)

if(PROTEUS_ENABLE_HIP)
  add_definitions("-DPROTEUS_ENABLE_HIP")
  find_package(hip REQUIRED CONFIG)
  message(STATUS "HIP Version: ${hip_VERSION}")
  if(hip_VERSION VERSION_LESS "5.7.1")
    message(FATAL_ERROR "HIP found: ${hip_VERSION} is less than minimum required version 5.7.1.")
  endif()
endif()

if(PROTEUS_ENABLE_CUDA)
  add_definitions("-DPROTEUS_ENABLE_CUDA")

  find_package(CUDAToolkit 12 REQUIRED)
endif()

if(PROTEUS_ENABLE_DEBUG)
  add_definitions("-DPROTEUS_ENABLE_DEBUG")
endif()

include(cmake/SetupLLVM.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/proteusConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(EXPORT proteusTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)

install(FILES
  "${PROJECT_BINARY_DIR}/proteusConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/proteusConfig.cmake"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/proteus)
