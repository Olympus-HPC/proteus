# NOTE: For CUDA, the proteus library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  JitEngine.cpp
  JitEngineHost.cpp
  Frontend/Func.cpp
  Frontend/Var.cpp
)

set(PROTEUS_PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")



if(PROTEUS_ENABLE_CUDA)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  JitEngineDeviceCUDA.cpp
)
endif()

if(PROTEUS_ENABLE_HIP)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  JitEngineDeviceHIP.cpp
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

target_compile_definitions(proteus PUBLIC ${LLVM_DEFINITIONS})
target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})
if(PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_HIP)
elseif(PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_CUDA)
endif()

if (PROTEUS_ENABLE_DEBUG)
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


if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(proteus USE_SHARED)
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
  #target_compile_options(proteus PRIVATE -x hip)
  #target_compile_options(proteusCore INTERFACE -x hip)
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

#target_link_libraries(proteus PRIVATE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
target_link_libraries(proteusCore INTERFACE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

set_target_properties(proteus PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

install(
  TARGETS proteus proteusCore
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Install headers preserving directory structure
install(DIRECTORY ${PROTEUS_PROJECT_INCLUDE_DIR}/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
