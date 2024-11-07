function(CREATE_CPU_TEST exe check_source)
    add_executable(${exe} ${check_source} ${ARGN})

    set_source_files_properties(${check_source} PROPERTIES
        OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/pass/ProteusPass.cpp
    )
    add_dependencies(${exe} ProteusPass)

    target_link_libraries(${exe} PUBLIC proteus)

    if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
        set(NATIVE_OPT_FLAGS "-mcpu=native" "-mtune=native")
    else()
        set(NATIVE_OPT_FLAGS "-march=native")
    endif()

    target_compile_options(
        ${exe}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusPass>
        ${NATIVE_OPT_FLAGS}
    )

    add_test(NAME ${exe} COMMAND ${LIT} -vv ${check_source})
    set_tests_properties(${exe} PROPERTIES FIXTURES_REQUIRED setupDeleteJitStoredCacheFiles)
    set_tests_properties(${exe} PROPERTIES FIXTURES_REQUIRED cleanupDeleteJitStoredCacheFiles)
endfunction()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
"
)

CREATE_CPU_TEST(daxpy daxpy.cpp)
CREATE_CPU_TEST(types types.cpp)
