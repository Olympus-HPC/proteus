set(SOURCES
  CompilerInterfaceHost.cpp
  CompilerInterfaceRuntimeConstantInfo.cpp
  CompiledLibrary.cpp
  Error.cpp
  JitEngine.cpp
  JitEngineHost.cpp
  Init.cpp
  TimeTracing.cpp
  Caching/ObjectCacheChain.cpp
  Caching/StorageCache.cpp
  Frontend/CppJitModule.cpp
  Frontend/Dispatcher.cpp
  Frontend/Func.cpp
  Frontend/JitFrontend.cpp
  Frontend/LoopUnroller.cpp
  Frontend/TargetModel.cpp
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

if(PROTEUS_ENABLE_MPI)
  list(APPEND SOURCES
    Caching/MPIHelpers.cpp
    Caching/MPIStorageCache.cpp
    Caching/MPILocalLookupCache.cpp
    Caching/MPIRemoteLookupCache.cpp
  )
endif()

if(BUILD_SHARED)
  add_library(proteus SHARED ${SOURCES})
else()
  add_library(proteus STATIC ${SOURCES})
endif()

add_library(proteusCore INTERFACE)
add_library(proteusFrontend INTERFACE)

target_compile_definitions(proteus PRIVATE ${LLVM_DEFINITIONS})
target_compile_definitions(proteusCore INTERFACE ${LLVM_DEFINITIONS})

if(PROTEUS_ENABLE_MPI)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_MPI)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_MPI)
endif()

if(PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_HIP)
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_HIP)
elseif(PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PUBLIC PROTEUS_ENABLE_CUDA)
  target_compile_definitions(proteus PRIVATE LIBDEVICE_BC_PATH="${LIBDEVICE_BC_FILE}")
  target_compile_definitions(proteus PRIVATE PROTEUS_CUDA_LIBDIR="${CUDAToolkit_LIBRARY_DIR}")
  target_compile_definitions(proteusCore INTERFACE PROTEUS_ENABLE_CUDA)
endif()

target_include_directories(proteus SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(proteusCore SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})

target_include_directories(proteus
  PUBLIC
  $<BUILD_INTERFACE:${PROTEUS_PROJECT_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${PROJECT_SOURCE_DIR}/src/include
)

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
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs
    Analysis
    BitReader
    BitWriter
    Core
    ExecutionEngine
    IPO
    IRReader
    Linker
    MC
    ${LLVM_TARGETS_TO_BUILD}
    Object
    OrcJit
    Passes
    Support
    Target
  )
endif()
message(STATUS "Found LLVM libraries: ${llvm_libs}")
target_link_libraries(proteus PRIVATE ${llvm_libs})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(proteusCore INTERFACE -fno-rtti)
endif()

if(PROTEUS_ENABLE_HIP)
  target_include_directories(proteus SYSTEM PRIVATE ${hip_INCLUDE_DIRS})
  target_link_libraries(proteus PRIVATE hip::host hiprtc::hiprtc)
  target_include_directories(proteusCore SYSTEM INTERFACE ${hip_INCLUDE_DIRS})

  # NOTE: HIP compilation (-x hip) is needed to include the HIP_SYMBOL macro,
  # which depends on the target architecture AMD or NVIDIA.  We define this
  # macro for HIP compilation in our headers to avoid forcing HIP compilation.
  # target_compile_options(proteus PRIVATE -x hip)
  # target_compile_options(proteusCore INTERFACE -x hip)


  if(NOT TARGET lldCommon)
    message(FATAL_ERROR "Missing lldCommon target. Check LLD cmake targets")
  endif()
  if(NOT TARGET lldELF)
    message(FATAL_ERROR "Missing lldELF target. Check LLD cmake targets")
  endif()

  set(lld_libs lldCommon lldELF)
  message(STATUS "Found LLD libraries: ${lld_libs}")
  target_link_libraries(proteus PRIVATE ${lld_libs})
endif()

if(PROTEUS_ENABLE_CUDA)
  add_library(proteus_cudart_builtins STATIC ProteusCUDARuntimeBuiltins.cpp)
  # Compiling builtins needs only the headers.
  target_include_directories(proteus_cudart_builtins PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(proteus_cudart_builtins PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(proteus INTERFACE proteus_cudart_builtins)

  # TODO: update cmake requirement to > 3.26 to supports the nvPTXCompiler target.
  # target_link_libraries(... CUDA::nvptxcompiler_static)
  find_library(LIBNVPTXCOMPILER
    NAMES nvptxcompiler nvptxcompiler_static libnvptxcompiler_static
    REQUIRED
    NO_DEFAULT_PATH
    PATHS ${CUDAToolkit_LIBRARY_DIR}
  )
  message(STATUS "Found libnvptxcompiler: ${LIBNVPTXCOMPILER}")

  target_include_directories(proteus SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  # Link just with the CUDA driver, leave the CUDA runtime to be linked by the
  # application static or shared. Proteus resolves its CUDA runtimes
  # dependencies at runtime through proteus builtins.
  target_link_libraries(proteus PUBLIC CUDA::cuda_driver)
  target_link_libraries(proteus PRIVATE "${LIBNVPTXCOMPILER}")
  target_include_directories(proteusCore SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

endif()

if(PROTEUS_ENABLE_MPI)
  target_link_libraries(proteus PRIVATE MPI::MPI_CXX)
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
message(STATUS "Found Clang libraries: ${clang_libs}")
target_link_libraries(proteus PRIVATE ${clang_libs})

# Add the clangxx binary path as a compile definition for use in the runtime,
# which needs it for the CPP frontend to resolve system path.
target_compile_definitions(proteus PRIVATE PROTEUS_CLANGXX_BIN="${CLANGXX}")

find_package(Threads REQUIRED)
target_link_libraries(proteus PRIVATE Threads::Threads)

if(BUILD_SHARED)
# Use a version script to hide internal symbols, especially from LLVM static
# libraries, on Linux. Only effective for shared proteus builds.
  if(UNIX AND NOT APPLE)
    target_link_options(proteus PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
  endif()
else()
  # When proteus is static and links LLVM/Clang/LLD statically, propagate
  # --exclude-libs for those specific archives so that consuming shared
  # libraries don't export duplicate LLVM global state.
  if(UNIX AND NOT APPLE)
    set(_exclude_archives "")

    if(NOT LLVM_LINK_LLVM_DYLIB)
      foreach(lib ${llvm_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(NOT CLANG_LINK_CLANG_DYLIB)
      foreach(lib ${clang_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(PROTEUS_ENABLE_HIP)
      foreach(lib ${lld_libs})
        list(APPEND _exclude_archives "lib${lib}.a")
      endforeach()
    endif()

    if(_exclude_archives)
      list(JOIN _exclude_archives ":" _exclude_libs_str)
      target_link_options(proteus INTERFACE
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-Wl,--exclude-libs,${_exclude_libs_str}>"
      )
    endif()
  endif()
endif()

target_link_libraries(proteusFrontend INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_link_libraries(proteusCore INTERFACE proteus "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

set_target_properties(proteus PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INTERFACE_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
)

set(PROTEUS_INSTALL_TARGETS proteus proteusCore proteusFrontend)
if(PROTEUS_ENABLE_CUDA)
    list(APPEND PROTEUS_INSTALL_TARGETS proteus_cudart_builtins)
endif()

install(
  TARGETS ${PROTEUS_INSTALL_TARGETS}
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Install headers preserving directory structure
install(DIRECTORY ${PROTEUS_PROJECT_INCLUDE_DIR}/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h"
)

if(PROTEUS_INSTALL_IMPL_HEADERS)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
  )
endif()
