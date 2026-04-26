function(CREATE_LLVMIR_CPU_LIT_TEST exe check_source)
  add_executable(${exe} ${check_source} ${ARGN})
  target_link_libraries(${exe} PUBLIC proteusFrontend)

  # Consumers need to add rdynamic when linking so that ORC JIT can link JIT
  # modules with exported symbols.
  target_link_options(${exe} PRIVATE $<LINK_ONLY:-rdynamic>)

  add_test(NAME ${exe}
           COMMAND ${LIT} -vv -D FILECHECK=${FILECHECK} -D LLVM_AS=${LLVM_AS}
                   ${check_source})
  set_property(TEST ${exe} PROPERTY LABELS "frontend-llvmir;frontend-llvmir-cpu")
endfunction()

find_program(LLVM_AS llvm-as
  PATHS ${LLVM_BINARY_DIR}/bin ${LLVM_TOOLS_BINARY_DIR})

if(NOT LLVM_AS)
  message(FATAL_ERROR "Testing requires llvm-as to be installed")
endif()

file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py
  "
import lit.formats
import os
import tempfile
import atexit
import shutil

config.name = 'LIT LLVMIR CPU tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
# Create a unique temp exec_root to avoid races on lit_test_times.txt
exec_root = tempfile.mkdtemp(prefix='lit.tmp.', dir='${CMAKE_CURRENT_BINARY_DIR}')
config.test_exec_root = exec_root
atexit.register(lambda: shutil.rmtree(exec_root, ignore_errors=False))

FILECHECK = lit_config.params['FILECHECK']
LLVM_AS = lit_config.params['LLVM_AS']
config.substitutions.append(('%FILECHECK', FILECHECK))
config.substitutions.append(('%llvm_as', LLVM_AS))
config.substitutions.append(('%build', '${CMAKE_CURRENT_BINARY_DIR}'))
")

set(LLVMIR_CPU_LIT_TESTS
    llvmir_jit_module_source
    llvmir_jit_module_bitcode)

foreach(exe IN LISTS LLVMIR_CPU_LIT_TESTS)
  CREATE_LLVMIR_CPU_LIT_TEST(${exe} ${exe}.cpp)
endforeach()
