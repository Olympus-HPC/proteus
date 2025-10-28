include(CMakeParseArguments)

# my_add_pass(<target> [ENABLE_AGGRESSIVE]
function(add_proteus target)
    set(options       ENABLE_AGGRESSIVE)
    cmake_parse_arguments(PROTEUS_PASS "${options}" "" "" ${ARGN})

    target_compile_options(${target} PRIVATE
        "-fpass-plugin=\$<TARGET_FILE:ProteusPass>"
    )

    if (PROTEUS_PASS_ENABLE_AGGRESSIVE)
	target_compile_options(${target} PRIVATE
	  "--mllvm -enable-aggressive-proteus-annotations>")
    endif()

    target_link_options(${target} PRIVATE
        "SHELL:\$<\$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=\$<TARGET_FILE:ProteusPass>>")

    get_target_property(target_type ${target} TYPE)
    if(NOT target_type)
        message(FATAL_ERROR "Target ${target} not found or does not have a TYPE property!")
    endif()

    get_target_property(proteus_target_type proteus TYPE)
    if(NOT proteus_target_type)
        message(FATAL_ERROR "Target proteus not found or does not have a TYPE property!")
    endif()

    # Do not link libproteus if it is a static library and the target is a
    # shared library, to avoid duplicating LLVM linking that causes static
    # initialization errors. Instead propagate the dependency of static
    # libproteus and its LLVM dependencies using INTERFACE.
    if(target_type STREQUAL "SHARED_LIBRARY" AND proteus_target_type STREQUAL "STATIC_LIBRARY")
        target_link_libraries(${target} INTERFACE -Wl,--allow-shlib-undefined proteus)
    else()
        target_link_libraries(${target} PRIVATE proteus)
    endif()
endfunction()
