cmake_minimum_required(VERSION 3.18)

project(ApolloJIT
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
option(ENABLE_HIP "Enable HIP" OFF)
option(ENABLE_DEBUG "Enable debugging output" OFF)
option(ENABLE_TESTS "Enable tests" ON)
set(LLVM_VERSION "17" CACHE STRING "LLVM version")

if(ENABLE_HIP)
  add_definitions("-DENABLE_HIP")
  find_package(hip 5.7.1 REQUIRED CONFIG)
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
