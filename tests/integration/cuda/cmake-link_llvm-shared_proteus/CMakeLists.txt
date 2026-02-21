cmake_minimum_required(VERSION 3.18)

project(
    main VERSION 0.0.1
    LANGUAGES CUDA CXX C)

find_package(CUDAToolkit REQUIRED)
find_package(proteus REQUIRED CONFIG HINTS ${proteus_DIR})

add_executable(main main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_link_options(main PRIVATE --no-undefined)
set_target_properties(main PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

add_proteus(main)

llvm_map_components_to_libnames(LLVM_LIBS Core ${LLVM_TARGETS_TO_BUILD} Support Target)
target_link_libraries(main PRIVATE ${LLVM_LIBS})
message(STATUS "Linked LLVM libs: ${LLVM_LIBS}")

install(TARGETS main)
