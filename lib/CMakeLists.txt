# NOTE: For CUDA, the proteus library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  JitEngine.cpp
  JitEngineHost.cpp
)

set(HEADERS
  CompilerInterfaceTypes.h
  JitCache.hpp
  JitEngine.hpp
  JitEngineHost.hpp
  JitStorageCache.hpp
  JitVariable.hpp
  TimeTracing.hpp
  TransformArgumentSpecialization.hpp
  Utils.h)

if(ENABLE_CUDA)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  CompilerInterfaceDeviceCUDA.cpp
  JitEngineDeviceCUDA.cpp
)

list(APPEND HEADERS
  CompilerInterfaceDevice.h
  JitEngineDevice.hpp
  JitEngineDeviceCUDA.hpp
  UtilsCUDA.h)
endif()

if(ENABLE_HIP)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  CompilerInterfaceDeviceHIP.cpp
  JitEngineDeviceHIP.cpp
)

list(APPEND HEADERS
  CompilerInterfaceDevice.h
  JitEngineDevice.hpp
  JitEngineDeviceHIP.hpp
  UtilsHIP.h)
endif()

if(BUILD_SHARED)
  if(ENABLE_CUDA)
    message(FATAL_ERROR "For CUDA, using libproteus as SHARED fails for device globals")
  endif()

  add_library(proteus SHARED ${SOURCES})
else()
  add_library(proteus STATIC ${SOURCES})
endif()

target_compile_definitions(proteus PRIVATE ${LLVM_DEFINITIONS})

if(ENABLE_TIME_TRACING)
  target_compile_definitions(proteus PRIVATE "-DENABLE_TIME_TRACING")
endif()

target_include_directories(proteus
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_include_directories(proteus
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
  $<INSTALL_INTERFACE:include>)

if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(proteus USE_SHARED)
else()
  set(libs ${LLVM_AVAILABLE_LIBS})
endif()

if(ENABLE_HIP)
  target_include_directories(proteus SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_compile_options(proteus PRIVATE -x hip)
  list(APPEND libs hip::host)

  # Link with LLD for non-RTC codegen.
  find_package(LLD REQUIRED NO_DEFAULT_PATH PATHS ${LLVM_INSTALL_DIR} CONFIG)
  message(STATUS "Found LLD DIR ${LLD_DIR}")
  list(APPEND libs lldCommon lldELF)
endif()

if(ENABLE_CUDA)
  target_include_directories(proteus SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  # list(APPEND libs CUDA::cuda_driver CUDA::cudart)
  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # list(APPEND libs CUDA::cuda_driver CUDA::cudart_static CUDA::nvptxcompiler_static)
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static
    ${CUDAToolkit_LIBRARY_DIR}/libnvptxcompiler_static.a)
endif()

target_link_libraries(proteus PRIVATE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
set_target_properties(proteus PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(
  TARGETS proteus
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(FILES
  ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Export proteusTargets to a CMake config file
install(
  EXPORT proteusTargets
  FILE proteusConfig.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/proteus"
)
