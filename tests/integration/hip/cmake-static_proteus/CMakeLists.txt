cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES HIP CXX C)

find_package(hip REQUIRED CONFIG)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

add_executable(main main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
target_link_options(main PRIVATE --no-undefined)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

add_proteus(main)

install(TARGETS main)
