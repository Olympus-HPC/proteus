# Point the tests at the just-built Python package in the build tree.
set(PROTEUS_PYTHON_TEST_ENV
  "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
)

# Always run the host-side C++ frontend tests.
add_test(
  NAME python_host_cpp_smoke
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_host_cpp_smoke.py
)
set_property(TEST python_host_cpp_smoke PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
set_property(TEST python_host_cpp_smoke PROPERTY LABELS "python")

add_test(
  NAME python_host_cpp_validation
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_host_cpp_validation.py
)
set_property(TEST python_host_cpp_validation PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
set_property(TEST python_host_cpp_validation PROPERTY LABELS "python")

if(PROTEUS_ENABLE_CUDA OR PROTEUS_ENABLE_HIP)
  # GPU C++ frontend coverage only makes sense when a device backend is enabled.
  add_test(
    NAME python_gpu_cpp_smoke
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_cpp_smoke.py
  )
  set_property(TEST python_gpu_cpp_smoke PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_cpp_smoke PROPERTY LABELS "python;frontend-gpu")

  add_test(
    NAME python_gpu_cpp_launch_validation
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_cpp_launch_validation.py
  )
  set_property(TEST python_gpu_cpp_launch_validation PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_cpp_launch_validation PROPERTY LABELS "python;frontend-gpu")

  add_test(
    NAME python_gpu_cpp_pointer_validation
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_cpp_pointer_validation.py
  )
  set_property(TEST python_gpu_cpp_pointer_validation PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_cpp_pointer_validation PROPERTY LABELS "python;frontend-gpu")
endif()

if(PROTEUS_ENABLE_MLIR AND (PROTEUS_ENABLE_CUDA OR PROTEUS_ENABLE_HIP))
  # MLIR coverage currently targets the GPU path, so require both features.
  add_test(
    NAME python_gpu_mlir_smoke
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_mlir_smoke.py
  )
  set_property(TEST python_gpu_mlir_smoke PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_mlir_smoke PROPERTY LABELS "python;frontend-gpu;frontend-mlir")

  add_test(
    NAME python_gpu_mlir_validation
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_mlir_validation.py
  )
  set_property(TEST python_gpu_mlir_validation PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_mlir_validation PROPERTY LABELS "python;frontend-gpu;frontend-mlir")

  add_test(
    NAME python_gpu_mlir_fatal_paths
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_mlir_fatal_paths.py
  )
  set_property(TEST python_gpu_mlir_fatal_paths PROPERTY ENVIRONMENT ${PROTEUS_PYTHON_TEST_ENV})
  set_property(TEST python_gpu_mlir_fatal_paths PROPERTY LABELS "python;frontend-gpu;frontend-mlir")
endif()
