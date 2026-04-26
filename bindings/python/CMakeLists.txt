# Build the Python extension against the interpreter chosen for this build.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The base bindings always expose the common Python API and the C++ frontend.
set(PROTEUS_PYTHON_SOURCES
  PythonBindings.cpp
  CppBindings.cpp
)
set(PROTEUS_PYTHON_PACKAGE_SOURCE_DIR
  "${PROJECT_SOURCE_DIR}/python/proteus")
set(PROTEUS_PYTHON_BUILD_PACKAGE_DIR
  "${CMAKE_BINARY_DIR}/python/proteus")
file(MAKE_DIRECTORY "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}")

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

configure_file(
  "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}/__init__.py"
  "${PROTEUS_PYTHON_BUILD_PACKAGE_DIR}/__init__.py"
  COPYONLY
)

if(PROTEUS_PYTHON_WHEEL)
  install(TARGETS proteus_python
    LIBRARY DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
  )
  install(FILES
    "${PROTEUS_PYTHON_PACKAGE_SOURCE_DIR}/__init__.py"
    DESTINATION "${PROTEUS_PYTHON_PACKAGE_DIR}"
  )
endif()

if(ENABLE_TESTS)
  # Keep the Python binding tests colocated with the extension target.
  add_subdirectory(tests)
endif()
