# NOTE: For CUDA, the proteus library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  CompilerInterfaceRuntimeConstantInfo.cpp
  JitEngine.cpp
  JitEngineHost.cpp
  Caching/StorageCache.cpp
  JitInterface.cpp
  Config.cpp
  Frontend/Func.cpp
  Frontend/VarStorage.cpp
  Frontend/Dispatcher.cpp
  Frontend/CppJitModule.cpp
)

set(PROTEUS_PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

if(PROTEUS_ENABLE_CUDA)
  list(APPEND SOURCES
    CompilerInterfaceDevice.cpp
    JitEngineDeviceCUDA.cpp
    Frontend/BuiltinsCUDA.cpp
  )
endif()

if(PROTEUS_ENABLE_HIP)
  list(APPEND SOURCES
    CompilerInterfaceDevice.cpp
    JitEngineDeviceHIP.cpp
    Frontend/BuiltinsHIP.cpp
  )
endif()

if(BUILD_SHARED)
  if(PROTEUS_ENABLE_CUDA)
    message(FATAL_ERROR "For CUDA, using libproteus as SHARED fails for device globals")
  endif()

  add_library(proteus SHARED ${SOURCES})
else()
  add_library(proteus STATIC ${SOURCES})
endif()

add_library(proteusCore INTERFACE)
add_library(proteusFrontend INTERFACE)

target_compile_definitions(proteusCore INTERFACE
  PROTEUS_ENABLE_HIP=$<IF:$<BOOL:${PROTEUS_ENABLE_HIP}>,1,0>
  PROTEUS_ENABLE_CUDA=$<IF:$<BOOL:${PROTEUS_ENABLE_CUDA}>,1,0>
)

target_compile_definitions(proteus PUBLIC ${LLVM_DEFINITIONS})

target_compile_definitions(proteus PUBLIC
  PROTEUS_ENABLE_HIP=$<IF:$<BOOL:${PROTEUS_ENABLE_HIP}>,1,0>
  PROTEUS_ENABLE_CUDA=$<IF:$<BOOL:${PROTEUS_ENABLE_CUDA}>,1,0>
)

target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

if(PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PRIVATE LIBDEVICE_BC_PATH="${LIBDEVICE_BC_FILE}")
  target_compile_definitions(proteus PRIVATE PROTEUS_CUDA_LIBDIR="${CUDAToolkit_LIBRARY_DIR}")
endif()

if(PROTEUS_ENABLE_TIME_TRACING)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_TIME_TRACING)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_TIME_TRACING)
endif()

target_include_directories(proteus
  SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(proteusCore
  SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})

target_include_directories(proteus
  PUBLIC
  $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(proteusCore
  INTERFACE
  $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link with LLVM libraries dynamically or statically based on the
# LLVM_LINK_LLVM_DYLIB option provided by the LLVM cmake setup.
if(LLVM_LINK_LLVM_DYLIB)
  if(NOT TARGET LLVM)
    message(FATAL_ERROR "Missing LLVM target. Check LLVM cmake targets")
  endif()
  set(libs LLVM)
else()
  set(libs ${LLVM_AVAILABLE_LIBS})
endif()

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(proteusCore INTERFACE -fno-rtti)
endif()

if(PROTEUS_ENABLE_HIP)
  target_include_directories(proteus SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_include_directories(proteusCore SYSTEM INTERFACE ${hip_INCLUDE_DIRS})

  # NOTE: HIP compilation (-x hip) is needed to include the HIP_SYMBOL macro,
  # which depends on the target architecture AMD or NVIDIA.  We define this
  # macro for HIP compilation in our headers to avoid forcing HIP compilation.
  # target_compile_options(proteus PRIVATE -x hip)
  # target_compile_options(proteusCore INTERFACE -x hip)
  list(APPEND libs hip::host)

  find_package(LLD REQUIRED CONFIG NO_DEFAULT_PATH
  HINTS "${LLVM_INSTALL_DIR}"
  PATH_SUFFIXES
    "lib/cmake/lld"
    "lib64/cmake/lld"
    "cmake/lld"
    "llvm/lib/cmake/lld"
    "llvm/lib64/cmake/lld"
  )

  if(NOT TARGET lldCommon)
    message(FATAL_ERROR "Missing lldCommon target. Check LLD cmake targets")
  endif()
  if(NOT TARGET lldELF)
    message(FATAL_ERROR "Missing lldELF target. Check LLD cmake targets")
  endif()

  list(APPEND libs lldCommon lldELF)
endif()

if(PROTEUS_ENABLE_CUDA)
  target_include_directories(proteus SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(proteusCore SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

  # list(APPEND libs CUDA::cuda_driver CUDA::cudart)
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static)
  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # list(APPEND libs CUDA::cuda_driver CUDA::cudart_static CUDA::nvptxcompiler_static)
  find_library(LIBNVPTXCOMPILER
    NAMES nvptxcompiler nvptxcompiler_static libnvptxcompiler_static
    REQUIRED
    NO_DEFAULT_PATH
    PATHS ${CUDAToolkit_LIBRARY_DIR}
  )
  list(APPEND libs "${LIBNVPTXCOMPILER}")
  message(STATUS "Found libnvptxcompiler: ${LIBNVPTXCOMPILER}")
endif()

find_program(CLANGXX
  NAMES clang++
  NO_DEFAULT_PATH
  REQUIRED
  PATHS
    "${LLVM_BINARY_DIR}/bin"
)
message(STATUS "Found clang++: ${CLANGXX}")

# Find required clang libraries in the LLVM installation.
if(CLANG_LINK_CLANG_DYLIB)
    list(APPEND clang_libs clang-cpp)
else()
  if(NOT TARGET clangFrontend)
    message(FATAL_ERROR "Missing clangCodeGen target. Check Clang cmake targets")
  endif()

  if(NOT TARGET clangCodeGen)
    message(FATAL_ERROR "Missing clangCodeGen target. Check Clang cmake targets")
  endif()

  list(APPEND clang_libs clangFrontend clangCodeGen)
endif()

# Print what we found for debugging
message(STATUS "Found LLVM libraries: ${libs}")
message(STATUS "Found Clang libraries: ${clang_libs}")

# Enclose linked libraries in a link group to avoid linker errors due to unresolved symbols.
target_link_libraries(proteus PUBLIC
  "$<$<PLATFORM_ID:Linux>:-Wl,--start-group>"
  ${libs} ${clang_libs}
  "$<$<PLATFORM_ID:Linux>:-Wl,--end-group>"
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
target_compile_definitions(proteus PRIVATE PROTEUS_CLANGXX_BIN="${CLANGXX}")

target_link_libraries(proteusFrontend INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_link_libraries(proteusCore INTERFACE proteus ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

set_target_properties(proteus PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

install(
  TARGETS proteus proteusCore proteusFrontend
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Install headers preserving directory structure
install(DIRECTORY ${PROTEUS_PROJECT_INCLUDE_DIR}/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
