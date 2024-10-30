# NOTE: For CUDA, the proteusjit library must be built as a static library to
# prevent symbol resolution issues with cudaGetSymbolAddress when linking device
# globals in the target application.
set(SOURCES
  CompilerInterfaceHost.cpp
  JitEngine.cpp
  JitEngineHost.cpp
)

if(ENABLE_CUDA)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  CompilerInterfaceDeviceCUDA.cpp
  JitEngineDeviceCUDA.cpp
)
endif()

if(ENABLE_HIP)
list(APPEND SOURCES
  CompilerInterfaceDevice.cpp
  CompilerInterfaceDeviceHIP.cpp
  JitEngineDeviceHIP.cpp
)
endif()

if(BUILD_SHARED_LIBJIT)
  if(ENABLE_CUDA)
    message(FATAL_ERROR "For CUDA, using libjit as SHARED fails for device globals")
  endif()

  add_library(proteusjit SHARED ${SOURCES})
else()
  add_library(proteusjit STATIC ${SOURCES})
endif()

target_compile_definitions(proteusjit PRIVATE ${LLVM_DEFINITIONS})

if(ENABLE_TIME_TRACING)
  target_compile_definitions(proteusjit PRIVATE "-DENABLE_TIME_TRACING")
endif()

target_include_directories(proteusjit
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(proteusjit USE_SHARED)
else()
  set(libs ${LLVM_AVAILABLE_LIBS})
endif()

if(ENABLE_HIP)
  target_include_directories(proteusjit SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_compile_options(proteusjit PRIVATE -x hip)
  list(APPEND libs hip::amdhip64)
endif()

if(ENABLE_CUDA)
  target_include_directories(proteusjit SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  # list(APPEND libs CUDA::cuda_driver CUDA::cudart)
  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # list(APPEND libs CUDA::cuda_driver CUDA::cudart_static CUDA::nvptxcompiler_static)
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static
    ${CUDAToolkit_LIBRARY_DIR}/libnvptxcompiler_static.a)
endif()

target_link_libraries(proteusjit PRIVATE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
set_target_properties(proteusjit PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS proteusjit)
