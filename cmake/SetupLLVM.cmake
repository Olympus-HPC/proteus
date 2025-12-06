# Find LLVM and Clang packages using the specified LLVM installation directory.
# The llvm/lib* suffixes are to support ROCm/LLVM installations.
find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
)
message(STATUS "Found LLVM package in: ${LLVM_DIR}")
message(STATUS "LLVM Version: ${LLVM_VERSION}")

find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_PREFIX}"
)
message(STATUS "Found Clang package in: ${Clang_DIR}")
message(STATUS "Clang Version: ${Clang_VERSION}")

if(PROTEUS_ENABLE_HIP)
  find_package(LLD REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_PREFIX}"
  )
  message(STATUS "Found LLD package in: ${LLD_DIR}")
  message(STATUS "LLD Version: ${LLD_VERSION}")
endif()

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
