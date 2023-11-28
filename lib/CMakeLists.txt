# NOTE: For CUDA the jit library is intentionally build as STATIC to avoid symbol
# resolution problems with cudaGetSymbolAddress for device globals linking on
# the targeted application
# NOTE : When SHARED, the jit library and the application to deploy on should
# link the CUDA runtime as a dynamic, shared library.
if(BUILD_SHARED_LIBJIT)
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
  list(APPEND libs CUDA::cuda_driver CUDA::cudart_static)
endif()

target_link_libraries(jit PRIVATE ${libs} "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
set_target_properties(jit PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS jit)
