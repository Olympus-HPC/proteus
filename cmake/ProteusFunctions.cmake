function(add_proteus target)
    set(options FORCE_JIT_ANNOTATE_ALL LINK_SHARED)
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
    if(NOT target_type)
        message(FATAL_ERROR "Target ${target} not found or does not have a TYPE property!")
    endif()

    set(proteus_lib_target proteus)
    if(PROTEUS_PASS_LINK_SHARED)
        if(NOT TARGET proteus_shared)
            message(FATAL_ERROR "add_proteus: LINK_SHARED requested but proteus_shared target not found. Ensure BUILD_SHARED=ON.")
        endif()
        set(proteus_lib_target proteus_shared)
    endif()

    get_target_property(proteus_target_type ${proteus_lib_target} TYPE)
    if(NOT proteus_target_type)
        message(FATAL_ERROR "Target ${proteus_lib_target} not found or does not have a TYPE property!")
    endif()

    # Do not link libproteus if it is a static library and the target is a
    # shared library, to avoid duplicating LLVM linking that causes static
    # initialization errors. Instead propagate the dependency of static
    # libproteus and its LLVM dependencies using INTERFACE.
    if(target_type STREQUAL "SHARED_LIBRARY" AND proteus_target_type STREQUAL "STATIC_LIBRARY")
        target_link_libraries(${target} INTERFACE -Wl,--allow-shlib-undefined proteus)
    else()
        target_link_libraries(${target} PRIVATE ${proteus_lib_target})
    endif()
endfunction()
