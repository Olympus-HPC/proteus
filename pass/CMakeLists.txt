# This creates an LLVM IR module that contains the type definition for
# proteus::RuntimeConstant.  It embeds the module in a header used by
# ProteusPass to match its definition of the RuntimeConstant type to the one
# expected on the target platform.
set(GEN_SOURCE_CODE "#include \"${CMAKE_SOURCE_DIR}/lib/CompilerInterfaceTypes.h\"\nstatic proteus::RuntimeConstant RC;")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake-gen.cpp "${GEN_SOURCE_CODE}")
set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake-out.ll)
set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/GenCompilerInterfaceTypes.h)

add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/cmake-gen.cpp -emit-llvm -S -o ${OUTPUT_FILE}
    DEPENDS ${PROJECT_SOURCE_DIR}/lib/CompilerInterfaceTypes.h
    COMMENT "Generating CompilerInterfaceTypes LLVM IR module file"
    VERBATIM
)

add_custom_command(
  OUTPUT ${HEADER_FILE}
  COMMAND ${CMAKE_COMMAND} -DOUTPUT_FILE=${OUTPUT_FILE} -DHEADER_FILE=${HEADER_FILE} -P ${CMAKE_CURRENT_BINARY_DIR}/embed_file.cmake
  DEPENDS ${OUTPUT_FILE}
  COMMENT "Generating ProteusPass header with GenModule"
)

add_custom_target(GenerateRuntimeConstantTyHeader DEPENDS ${HEADER_FILE})

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/embed_file.cmake [[
file(READ "${OUTPUT_FILE}" FILE_CONTENTS)
file(WRITE "${HEADER_FILE}" "static const char GenModule[] = R\"===(${FILE_CONTENTS})===\";")
]])

add_library(ProteusPass SHARED ProteusPass.cpp)

add_dependencies(ProteusPass GenerateRuntimeConstantTyHeader)
target_include_directories(ProteusPass
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(ProteusPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ProteusPass PRIVATE -fno-rtti)
endif()

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(ProteusPass
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_compile_options(
  ProteusPass
  INTERFACE
  -fpass-plugin=$<TARGET_FILE:ProteusPass>
)

target_link_options(
  ProteusPass
  INTERFACE
  "SHELL:$<$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=$<TARGET_FILE:ProteusPass>>"
)

install(
  TARGETS ProteusPass
  EXPORT proteusTargets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
