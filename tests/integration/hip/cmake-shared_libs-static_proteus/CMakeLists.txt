cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES HIP CXX C)

find_package(hip REQUIRED CONFIG)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

# libfoo
set_source_files_properties(libfoo.cpp PROPERTIES LANGUAGE HIP)
add_library(libfoo SHARED libfoo.cpp)
target_compile_options(libfoo PRIVATE -fgpu-rdc)
target_link_options(libfoo
    PRIVATE
    --no-undefined
    -fgpu-rdc
    --hip-link
)
target_link_libraries(libfoo PUBLIC hip::device)
add_proteus(libfoo)

# libbar
set_source_files_properties(libbar.cpp PROPERTIES LANGUAGE HIP)
add_library(libbar SHARED libbar.cpp)
target_compile_options(libbar PRIVATE -fgpu-rdc)
target_link_options(libbar
    PRIVATE
    --no-undefined
    -fgpu-rdc
    --hip-link
)
target_link_libraries(libbar PUBLIC hip::device)
add_proteus(libbar)

# main
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(main main.cpp)
target_compile_options(main
    PRIVATE
    -fgpu-rdc
)
target_link_options(main
    PRIVATE
    --no-undefined
    -fgpu-rdc
    --hip-link
)
target_link_libraries(main PUBLIC libfoo libbar)
add_proteus(main)

install(TARGETS main libfoo libbar)
