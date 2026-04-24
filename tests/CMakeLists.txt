find_program(LIT lit
    PATHS ${LLVM_BINARY_DIR}/bin)

if(NOT LIT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import lit"
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_QUIET
    )

    if(${EXIT_CODE} EQUAL 0)
        set(LIT ${CMAKE_CURRENT_SOURCE_DIR}/lit-main.py)
        message(STATUS "Using ${LIT} for lit testing")
    else()
        message(FATAL_ERROR "Testing requires lit to be installed, e.g., python -m pip install lit>=15.0.7")
    endif()
else()
    message(STATUS "Found lit at ${LIT}")
endif()

find_program(FILECHECK FileCheck
    PATHS ${LLVM_BINARY_DIR}/bin ${LLVM_BINARY_DIR}/libexec/llvm)

if(NOT FILECHECK)
    message(FATAL_ERROR "Testing requires FileCheck to be installed")
else()
    message(STATUS "Found Filecheck at ${FILECHECK}")
endif()

function(proteus_attach_pass_plugin_rebuild_dep target)
    set(proteus_pass_stamp "${CMAKE_CURRENT_BINARY_DIR}/${target}.proteuspass.stamp")

    # Rebuild test sources when the pass plugin is rebuilt, since the compiler
    # consumes the plugin during compilation via -fpass-plugin.
    add_custom_command(
        OUTPUT "${proteus_pass_stamp}"
        COMMAND ${CMAKE_COMMAND} -E touch "${proteus_pass_stamp}"
        DEPENDS ProteusPass
        VERBATIM
    )

    set_source_files_properties(${ARGN} PROPERTIES
        OBJECT_DEPENDS "${proteus_pass_stamp}"
    )
    target_compile_options(
        ${target}
        PUBLIC
        -fpass-plugin=$<TARGET_FILE:ProteusPass>
    )
endfunction()

add_subdirectory(cpu)

if(PROTEUS_ENABLE_HIP OR PROTEUS_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()

add_subdirectory(frontend)
