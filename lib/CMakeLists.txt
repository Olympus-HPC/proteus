# NOTE: For CUDA the jit library is must build as STATIC to avoid symbol
# resolution problems with cudaGetSymbolAddress for device globals linking on
# the targeted application
if(BUILD_SHARED_LIBJIT)
  if(ENABLE_CUDA)
    message(FATAL_ERROR "For CUDA, using libjit as SHARED fails for device globals")
  endif()

  add_library(jit SHARED jit.cpp)
else()
  add_library(jit STATIC jit.cpp)
endif()

target_compile_definitions(jit PRIVATE ${LLVM_DEFINITIONS})

if(ENABLE_RUNTIME_CONSTPROP)
  target_compile_definitions(jit PRIVATE "-DENABLE_RUNTIME_CONSTPROP")
endif()

if(ENABLE_JIT_LAUNCH_BOUNDS)
  target_compile_definitions(jit PRIVATE "-DENABLE_JIT_LAUNCH_BOUNDS")
endif()

if(ENABLE_TIME_TRACING)
  target_compile_definitions(jit PRIVATE "-DENABLE_TIME_TRACING")
endif()

target_include_directories(jit
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(jit USE_SHARED)
else()
  set(libs ${LLVM_AVAILABLE_LIBS})
endif()

if(ENABLE_HIP)
  target_include_directories(jit SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_compile_options(jit PRIVATE -x hip)
  list(APPEND libs hip::amdhip64)
endif()

if(ENABLE_CUDA)
  target_include_directories(jit SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

  # list(APPEND libs CUDA::cuda_driver CUDA::cudart)
  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # list(APPEND libs CUDA::cuda_driver CUDA::cudart_static CUDA::nvptxcompiler_static)
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static
    ${CUDAToolkit_LIBRARY_DIR}/libnvptxcompiler_static.a)
endif()

target_link_libraries(jit PRIVATE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
set_target_properties(jit PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS jit)
