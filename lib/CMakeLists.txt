add_library(jit SHARED jit.cpp)

target_compile_definitions(jit PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(jit 
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

llvm_map_components_to_libnames(llvm_libs core orcjit native)
target_link_libraries(jit ${llvm_libs})