# NOTE: For CUDA, the proteus library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  CompilerInterfaceRuntimeConstantInfo.cpp
  CompiledLibrary.cpp
  Error.cpp
  JitEngine.cpp
  JitEngineHost.cpp
  TimeTracing.cpp
  Caching/ObjectCacheChain.cpp
  Caching/StorageCache.cpp
  Frontend/CppJitModule.cpp
  Frontend/Dispatcher.cpp
  Frontend/Func.cpp
  Frontend/JitFrontend.cpp
  Frontend/LoopUnroller.cpp
  Frontend/TypeMap.cpp
  Frontend/VarStorage.cpp
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

add_library(proteus STATIC ${SOURCES})
set(PROTEUS_TARGETS proteus)

if(BUILD_SHARED)
  add_library(proteus_shared SHARED ${SOURCES})
  if(PROTEUS_ENABLE_CUDA)
    set_target_properties(proteus_shared PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  endif()
  set_target_properties(proteus_shared PROPERTIES OUTPUT_NAME proteus)
  list(APPEND PROTEUS_TARGETS proteus_shared)
endif()

add_library(proteusCore INTERFACE)
add_library(proteusFrontend INTERFACE)

foreach(proteus_target ${PROTEUS_TARGETS})
  target_compile_definitions(${proteus_target} PUBLIC ${LLVM_DEFINITIONS})
endforeach()
target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

if(PROTEUS_ENABLE_HIP)
  foreach(proteus_target ${PROTEUS_TARGETS})
    target_compile_definitions(${proteus_target} PUBLIC PROTEUS_ENABLE_HIP)
  endforeach()
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_HIP)
elseif(PROTEUS_ENABLE_CUDA)
  foreach(proteus_target ${PROTEUS_TARGETS})
    target_compile_definitions(${proteus_target} PUBLIC PROTEUS_ENABLE_CUDA)
    target_compile_definitions(${proteus_target} PRIVATE LIBDEVICE_BC_PATH="${LIBDEVICE_BC_FILE}")
    target_compile_definitions(${proteus_target} PRIVATE PROTEUS_CUDA_LIBDIR="${CUDAToolkit_LIBRARY_DIR}")
  endforeach()
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_CUDA)
endif()

foreach(proteus_target ${PROTEUS_TARGETS})
  target_include_directories(${proteus_target}
    SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
endforeach()
target_include_directories(proteusCore
  SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})

foreach(proteus_target ${PROTEUS_TARGETS})
  target_include_directories(${proteus_target}
    PUBLIC
    $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endforeach()

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
  foreach(proteus_target ${PROTEUS_TARGETS})
    target_include_directories(${proteus_target} SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  endforeach()
  target_include_directories(proteusCore SYSTEM INTERFACE ${hip_INCLUDE_DIRS})

  # NOTE: HIP compilation (-x hip) is needed to include the HIP_SYMBOL macro,
  # which depends on the target architecture AMD or NVIDIA.  We define this
  # macro for HIP compilation in our headers to avoid forcing HIP compilation.
  # target_compile_options(proteus PRIVATE -x hip)
  # target_compile_options(proteusCore INTERFACE -x hip)
  list(APPEND libs hip::host hiprtc::hiprtc)


  if(NOT TARGET lldCommon)
    message(FATAL_ERROR "Missing lldCommon target. Check LLD cmake targets")
  endif()
  if(NOT TARGET lldELF)
    message(FATAL_ERROR "Missing lldELF target. Check LLD cmake targets")
  endif()

  list(APPEND libs lldCommon lldELF)
endif()

if(PROTEUS_ENABLE_CUDA)
  foreach(proteus_target ${PROTEUS_TARGETS})
    target_include_directories(${proteus_target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endforeach()
  target_include_directories(proteusCore SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

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
foreach(proteus_target ${PROTEUS_TARGETS})
  set(device_rt_libs "")
  if(PROTEUS_ENABLE_CUDA)
    list(APPEND device_rt_libs CUDA::cuda_driver)
    if(proteus_target STREQUAL "proteus_shared")
      list(APPEND device_rt_libs CUDA::cudart) 
    else()
      list(APPEND device_rt_libs CUDA::cudart_static) 
    endif()
  endif()
  target_link_libraries(${proteus_target} PUBLIC
    "$<$<PLATFORM_ID:Linux>:-Wl,--start-group>"
    ${device_rt_libs} ${libs} ${clang_libs}
    "$<$<PLATFORM_ID:Linux>:-Wl,--end-group>"
    "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
  )
  target_compile_definitions(${proteus_target} PRIVATE PROTEUS_CLANGXX_BIN="${CLANGXX}")
endforeach()

target_link_libraries(proteusFrontend INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_link_libraries(proteusCore INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

foreach(proteus_target ${PROTEUS_TARGETS})
  set_target_properties(${proteus_target} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
endforeach()

install(
  TARGETS ${PROTEUS_TARGETS} proteusCore proteusFrontend
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Install headers preserving directory structure
install(DIRECTORY ${PROTEUS_PROJECT_INCLUDE_DIR}/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h"
)
