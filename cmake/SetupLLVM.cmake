# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang/")
set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm")
set(Clang_DIR "${LLVM_INSTALL_DIR}/lib/cmake/clang")

find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH)

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH)
