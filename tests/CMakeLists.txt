find_program(LIT lit
    PATHS ${LLVM_INSTALL_DIR}/bin)

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
    PATHS ${LLVM_INSTALL_DIR}/bin ${LLVM_INSTALL_DIR}/libexec/llvm)

if(NOT FILECHECK)
    message(FATAL_ERROR "Testing requires FileCheck to be installed")
else()
    message(STATUS "Found Filecheck at ${FILECHECK}")
endif()

add_subdirectory(cpu)

if(PROTEUS_ENABLE_HIP OR PROTEUS_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()

add_subdirectory(frontend)
