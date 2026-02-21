function(add_proteus target)
    set(options FORCE_JIT_ANNOTATE_ALL)
    cmake_parse_arguments(PROTEUS_PASS "${options}" "" "" ${ARGN})

    if (PROTEUS_PASS_FORCE_JIT_ANNOTATE_ALL)
        target_compile_options(${target} PRIVATE
            # -fplugin loading is needed to register the plugin command line
            # option.
            "-fplugin=$<TARGET_FILE:ProteusPass>"
            "-fpass-plugin=$<TARGET_FILE:ProteusPass>"
            "SHELL:-Xclang -mllvm -Xclang -force-proteus-jit-annotate-all"
        )
    else()
        target_compile_options(${target} PRIVATE
            "-fpass-plugin=\$<TARGET_FILE:ProteusPass>"
        )
    endif()

    target_link_options(${target} PRIVATE
        "SHELL:\$<\$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=\$<TARGET_FILE:ProteusPass>>")

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        target_link_libraries(${target} PRIVATE proteus)
    else()
        target_link_libraries(${target} PUBLIC proteus)
    endif()

    message(STATUS "Linked target '${target}' with libproteus")
endfunction()
