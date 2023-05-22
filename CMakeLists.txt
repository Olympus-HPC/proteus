cmake_minimum_required(VERSION 3.18)

project(ApolloJIT
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")

include(cmake/SetupLLVM.cmake)

add_subdirectory(lib)
add_subdirectory(pass)

add_subdirectory(tests)