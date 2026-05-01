# Build the Python extension against the interpreter chosen for this build.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The base bindings always expose the common Python API and the C++ frontend.
set(PROTEUS_PYTHON_SOURCES
  PythonBindings.cpp
  CppBindings.cpp
  LLVMIRBindings.cpp
)
set(PROTEUS_PYTHON_BUILD_PACKAGE_DIR
  "${CMAKE_BINARY_DIR}/python/${PROTEUS_PYTHON_PACKAGE_DIR}")
file(MAKE_DIRECTORY "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}")
file(COPY "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/python/${PROTEUS_PYTHON_PACKAGE_DIR}")

string(TOLOWER "${CMAKE_SYSTEM_NAME}" PROTEUS_PYTHON_PLATFORM_SYSTEM)
set(PROTEUS_PYTHON_VARIANT_PLATFORM
  "${PROTEUS_PYTHON_PLATFORM_SYSTEM}-${CMAKE_SYSTEM_PROCESSOR}")
set(PROTEUS_PYTHON_VARIANT_LLVM_MAJOR "${LLVM_VERSION_MAJOR}")
set(PROTEUS_PYTHON_VARIANT_CUDA_MAJOR "null")
set(PROTEUS_PYTHON_VARIANT_ROCM_VERSION "null")

if(PROTEUS_ENABLE_CUDA)
  set(PROTEUS_PYTHON_VARIANT_KIND "cuda")
  set(PROTEUS_PYTHON_VARIANT_ID "cuda${CUDAToolkit_VERSION_MAJOR}_llvm${LLVM_VERSION_MAJOR}")
  set(PROTEUS_PYTHON_VARIANT_CUDA_MAJOR "${CUDAToolkit_VERSION_MAJOR}")
  set(PROTEUS_PYTHON_VARIANT_PRIORITY 1200)
elseif(PROTEUS_ENABLE_HIP)
  set(PROTEUS_PYTHON_VARIANT_KIND "rocm")
  if(hip_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
    set(PROTEUS_PYTHON_VARIANT_ID "rocm${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    set(PROTEUS_PYTHON_VARIANT_ROCM_VERSION "\"${CMAKE_MATCH_1}.${CMAKE_MATCH_2}\"")
  else()
    set(PROTEUS_PYTHON_VARIANT_ID "rocm")
  endif()
  set(PROTEUS_PYTHON_VARIANT_PRIORITY 700)
elseif(PROTEUS_PYTHON_PACKAGE_DIR STREQUAL "proteus")
  set(PROTEUS_PYTHON_VARIANT_KIND "shim")
  set(PROTEUS_PYTHON_VARIANT_ID "shim")
  set(PROTEUS_PYTHON_VARIANT_PRIORITY 0)
else()
  set(PROTEUS_PYTHON_VARIANT_KIND "host")
  set(PROTEUS_PYTHON_VARIANT_ID "host_llvm${LLVM_VERSION_MAJOR}")
  set(PROTEUS_PYTHON_VARIANT_PRIORITY 100)
endif()

if(PROTEUS_ENABLE_MLIR)
  # Add the MLIR entrypoint only when that backend is available in the build.
  list(APPEND PROTEUS_PYTHON_SOURCES MLIRBindings.cpp)
endif()

# Produce the importable extension module that backs `import proteus`.
pybind11_add_module(proteus_python MODULE ${PROTEUS_PYTHON_SOURCES})
target_link_libraries(proteus_python PRIVATE proteusFrontend)

# Mirror the selected backend features into the binding compilation unit.
target_compile_definitions(proteus_python PRIVATE
  $<$<BOOL:${PROTEUS_ENABLE_MLIR}>:PROTEUS_ENABLE_MLIR>
  $<$<BOOL:${PROTEUS_ENABLE_CUDA}>:PROTEUS_ENABLE_CUDA>
  $<$<BOOL:${PROTEUS_ENABLE_HIP}>:PROTEUS_ENABLE_HIP>
)

# Emit the extension as `proteus.*` in a dedicated build-tree Python package dir.
set_target_properties(proteus_python PROPERTIES
  OUTPUT_NAME _proteus
  LIBRARY_OUTPUT_DIRECTORY "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}"
  INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
)

if(EXISTS "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}/manifest.json.in")
  configure_file(
    "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}/manifest.json.in"
    "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}/manifest.json"
    @ONLY
  )
endif()

if(PROTEUS_PYTHON_WHEEL)
  install(TARGETS proteus_python
    LIBRARY DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
  )
  install(DIRECTORY
    "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}/"
    DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "manifest.json.in" EXCLUDE
    PATTERN "_proteus*.so" EXCLUDE
    PATTERN "_proteus*.dylib" EXCLUDE
  )
endif()

if(ENABLE_TESTS)
  # Keep the Python binding tests colocated with the extension target.
  add_subdirectory(tests)
endif()
