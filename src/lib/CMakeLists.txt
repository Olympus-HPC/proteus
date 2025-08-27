# NOTE: For CUDA, the proteus library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  CompilerInterfaceRuntimeConstantInfo.cpp
  JitEngine.cpp
  JitEngineHost.cpp
  Frontend/Func.cpp
  Frontend/Var.cpp
  Frontend/Array.cpp
  Frontend/Dispatcher.cpp
  Frontend/CppJitModule.cpp
  Frontend/LoopNest.cpp
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

target_compile_definitions(proteus PUBLIC ${LLVM_DEFINITIONS})
target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

if(PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_HIP)
elseif(PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PRIVATE LIBDEVICE_BC_PATH="${LIBDEVICE_BC_FILE}")
  target_compile_definitions(proteus PRIVATE PROTEUS_CUDA_LIBDIR="${CUDAToolkit_LIBRARY_DIR}")
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_CUDA)
endif()

if(PROTEUS_ENABLE_DEBUG)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_DEBUG)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_DEBUG)
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

if(PROTEUS_LINK_SHARED_LLVM)
  if(LLVM_LINK_LLVM_DYLIB)
    llvm_config(proteus USE_SHARED)
    set(libs LLVM)
  else()
    find_library(LLVM_SHARED_LIB
      NAMES LLVM
      PATHS "${LLVM_LIBRARY_DIR}"
      NO_DEFAULT_PATH
    )

    if(LLVM_SHARED_LIB)
      message(STATUS "Found libLLVM.so: ${LLVM_SHARED_LIB}")
      set(libs ${LLVM_SHARED_LIB})
    else()
      message(FATAL_ERROR "The LLVM installation at ${LLVM_DIR} does not provide libLLVM.so, required due to "
        "PROTEUS_LINK_SHARED_LLVM=ON.
      Change PROTEUS_LINK_SHARED_LLVM to OFF.
      ")
    endif()
  endif()
else()
  if(LLVM_LINK_LLVM_DYLIB)
    message(FATAL_ERROR "The LLVM installation at ${LLVM_DIR} requires to link with "
      "the LLVM shared library, but PROTEUS_LINK_SHARED_LLVM=OFF.
    Change PROTEUS_LINK_SHARED_LLVM to ON.")
  endif()

  set(libs ${LLVM_AVAILABLE_LIBS})

  # Some LLVM installations include in the LLVM_AVAILABLE_LIBS both an LLVM
  # shared library AND static libraries. The LLVM shared library bundles those
  # static libraries and we have to remove it to avoid static initialization
  # errors from linking with both.
  list(REMOVE_ITEM libs "LLVM")
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

  # Link with LLD libraries for non-RTC codegen. Use explicit location to avoid
  # importing from system directories.
  list(APPEND libs ${LLVM_INSTALL_DIR}/lib/liblldCommon.a ${LLVM_INSTALL_DIR}/lib/liblldELF.a)
endif()

if(PROTEUS_ENABLE_CUDA)
  target_include_directories(proteus SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(proteusCore SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

  # list(APPEND libs CUDA::cuda_driver CUDA::cudart)
  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # list(APPEND libs CUDA::cuda_driver CUDA::cudart_static CUDA::nvptxcompiler_static)
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static
    ${CUDAToolkit_LIBRARY_DIR}/libnvptxcompiler_static.a)
endif()

find_program(CLANGXX
  NAMES clang++
  PATHS "${LLVM_INSTALL_DIR}/bin"
  NO_DEFAULT_PATH
  REQUIRED
)
message(STATUS "Found clang++: ${CLANGXX}")

# Find all libclang*.a libraries in the LLVM installation
file(GLOB clang_libs "${LLVM_INSTALL_DIR}/lib/libclang*.a")

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

target_link_libraries(proteusCore INTERFACE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

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
