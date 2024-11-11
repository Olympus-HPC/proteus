
SET(sources main.cpp simulation.cpp io.cpp init.cpp material.cpp utils.cpp)
set(original hip-rsbench)
set(jit_version hip-jit-rsbench)

add_executable(${original} ${sources})
set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
set_target_properties(${original} PROPERTIES HIP_ARCHITECTURES gfx90a)
target_include_directories(${original} PUBLIC ${hip_INCLUDE_DIRS})
#message(WARNING "DIRS ARE ${hip_INCLUDE_DIRS}")
#target_compile_options()
target_link_libraries(${original} PUBLIC hip::host hip::device)

#add_executable(${jit_version} ${sources})
#set_target_properties(${jit_version} PROPERTIES HIP_ARCHITECTURES gfx90a)
#target_link_libraries(${jit_version} PUBLIC hip::host hip::device)


#target_link_libraries(${jit_version} PUBLIC jit)
#target_compile_definitions(${jit_version} PUBLIC "__JIT_IS_AWESOME__")
#target_link_directories(${jit_version} PRIVATE "${JIT_ROOT}/lib/")
#target_compile_options(${jit_version}
#    PUBLIC
#    -fpass-plugin=${JIT_ROOT}/pass/libProteusJitPass.so)

