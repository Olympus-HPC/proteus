# Find LLVM and Clang packages using the specified LLVM installation directory.
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES "lib/cmake/llvm" "lib64/cmake/llvm" "cmake/llvm"
)

find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES "lib/cmake/clang" "lib64/cmake/clang" "cmake/clang"
)

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
