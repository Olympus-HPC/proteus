if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
    endif()

enable_language(CUDA)
message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
    message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
endif()

function(CREATE_FRONTEND_CUDA_TEST exe check_source)
    add_executable(${exe}.CUDA ${check_source} ${ARGN})
    set_source_files_properties(${check_source} ${ARGN} PROPERTIES
        LANGUAGE CUDA
    )
    add_dependencies(${exe}.CUDA ProteusPass)

    target_link_libraries(${exe}.CUDA PUBLIC proteus)

    # set_target_properties(${exe}.${lang} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME ${exe}.CUDA COMMAND ${LIT} -vv -D EXT=CUDA -DFILECHECK=${FILECHECK} ${check_source})
endfunction()

CREATE_FRONTEND_CUDA_TEST(assign assign.cpp)
