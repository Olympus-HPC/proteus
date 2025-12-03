cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES HIP CXX C)

set(CMAKE_CXX_STANDARD 17)
find_package(hip REQUIRED CONFIG)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

# Build the executable.
add_executable(main main.cpp)
target_link_libraries(main PUBLIC proteusFrontend hip::host)

target_link_options(main PRIVATE
    --no-undefined
    $<LINK_ONLY:-rdynamic>
)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Build the clang plugin library.
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
add_library(ClangPlugin SHARED ClangPlugin.cpp)
target_include_directories(ClangPlugin PRIVATE "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS}")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

set_target_properties(ClangPlugin PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS main ClangPlugin
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "bin"
)
