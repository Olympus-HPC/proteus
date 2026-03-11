function(CREATE_MLIR_CPU_LIT_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  target_link_libraries(${exe} PUBLIC proteusFrontend)

  # Consumers need to add rdynamic when linking so that ORC JIT can link JIT
  # modules with exported symbols.
  target_link_options(${exe} PRIVATE $<LINK_ONLY:-rdynamic>)

  add_test(NAME ${exe}
           COMMAND ${LIT} -vv -D FILECHECK=${FILECHECK} ${check_source})
  set_property(TEST ${exe} PROPERTY LABELS "frontend-mlir;frontend-mlir-cpu")
endfunction()

file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT MLIR CPU tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
# Create a unique temp exec_root to avoid races on lit_test_times.txt
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
")

set(MLIR_CPU_LIT_TESTS
    mlir_add_vectors
    mlir_add_vectors_runconst
    mlir_array
    mlir_atomics
    mlir_calls
    mlir_casts
    mlir_cmp
    mlir_convert
    mlir_external_call
    mlir_for
    mlir_for_unroll
    mlir_functional_blocks
    mlir_generic_load_store
    mlir_get_address
    mlir_host_jit_execute
    mlir_if
    mlir_int_atomics
    mlir_internal_call
    mlir_intrinsics
    mlir_loopnest_1d
    mlir_loopnest_3d
    mlir_loopnest_3d_uniformtile
    mlir_min
    mlir_multi_module
    mlir_operators
    mlir_pointer_reassign
    mlir_pointer_reassign_dynamic
    mlir_scalar_add
    mlir_tiled_matmul
    mlir_tiled_transpose
    mlir_vars
    mlir_while)

foreach(exe IN LISTS MLIR_CPU_LIT_TESTS)
  CREATE_MLIR_CPU_LIT_TEST(${exe} ${exe}.cpp)
endforeach()
