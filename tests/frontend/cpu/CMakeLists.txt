function(CREATE_FRONTEND_TEST exe check_source)
    add_executable(${exe} ${check_source} ${ARGN})

    target_link_libraries(${exe} PUBLIC proteusFrontend)

    if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
        set(NATIVE_OPT_FLAGS "-mcpu=native" "-mtune=native")
    else()
        set(NATIVE_OPT_FLAGS "-march=native")
    endif()

    target_compile_options(
        ${exe}
        PUBLIC
        ${NATIVE_OPT_FLAGS}
    )

    # Consumers need to add rdynamic when linking so that ORC JIT can link JIT
    # modules with exported symbols.
    target_link_options(${exe} PRIVATE $<LINK_ONLY:-rdynamic>)

    add_test(NAME ${exe} COMMAND ${LIT} -vv -D FILECHECK=${FILECHECK} ${check_source})
    set_property(TEST ${exe} PROPERTY LABELS "frontend-cpu")
endfunction()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats
import os

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)
config.environment = os.environ.copy()

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
FILECHECK = lit_config.params['FILECHECK']
config.substitutions.append(('%FILECHECK', FILECHECK))
"
)

CREATE_FRONTEND_TEST(operators operators.cpp)
CREATE_FRONTEND_TEST(if if.cpp)
CREATE_FRONTEND_TEST(for for.cpp)
CREATE_FRONTEND_TEST(external_call external_call.cpp)
CREATE_FRONTEND_TEST(add_vectors add_vectors.cpp)
