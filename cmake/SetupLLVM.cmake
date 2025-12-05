# Find LLVM and Clang packages using the specified LLVM installation directory.
# The llvm/lib* suffixes are to support ROCm/LLVM installations.
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES
    "lib/cmake/llvm"
    "lib64/cmake/llvm"
    "cmake/llvm"
    "llvm/lib/cmake/llvm"
    "llvm/lib64/cmake/llvm"
)

find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES
    "lib/cmake/clang"
    "lib64/cmake/clang"
    "cmake/clang"
    "llvm/lib/cmake/clang"
    "llvm/lib64/cmake/clang"
)

find_package(LLD REQUIRED CONFIG NO_DEFAULT_PATH
HINTS "${LLVM_INSTALL_DIR}"
PATH_SUFFIXES
  "lib/cmake/lld"
  "lib64/cmake/lld"
  "cmake/lld"
  "llvm/lib/cmake/lld"
  "llvm/lib64/cmake/lld"
)

message(STATUS "Found LLVM package in: ${LLVM_DIR}")
message(STATUS "Found Clang package in: ${Clang_DIR}")
message(STATUS "Found LLD package in: ${LLD_DIR}")
message(STATUS "LLVM Version: ${LLVM_VERSION}")
message(STATUS "Clang Version: ${Clang_VERSION}")
message(STATUS "LLD Version: ${LLD_VERSION}")

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
