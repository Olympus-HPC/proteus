function(CREATE_MLIR_FRONTEND_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  target_link_libraries(${exe} PUBLIC proteusFrontend)

  if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
    set(NATIVE_OPT_FLAGS "-mcpu=native" "-mtune=native")
  else()
    set(NATIVE_OPT_FLAGS "-march=native")
  endif()

  target_compile_options(${exe} PUBLIC ${NATIVE_OPT_FLAGS})
  target_link_options(${exe} PRIVATE $<LINK_ONLY:-rdynamic>)

  add_test(NAME ${exe} COMMAND ${LIT} -vv -D FILECHECK=${FILECHECK} ${check_source})
  set_property(TEST ${exe} PROPERTY LABELS "frontend-mlir")
endfunction()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT MLIR tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
"
)

CREATE_MLIR_FRONTEND_TEST(mlir_scalar_add mlir_scalar_add.cpp)
CREATE_MLIR_FRONTEND_TEST(mlir_if mlir_if.cpp)
CREATE_MLIR_FRONTEND_TEST(mlir_for mlir_for.cpp)
CREATE_MLIR_FRONTEND_TEST(mlir_while mlir_while.cpp)
CREATE_MLIR_FRONTEND_TEST(mlir_cmp mlir_cmp.cpp)
