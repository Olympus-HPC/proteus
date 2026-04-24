# We use the full path because we query those source files to make them
# dependencies for building tests.
set(PROTEUS_PASS_SOURCES
  ${PROJECT_SOURCE_DIR}/src/pass/AnnotationHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/pass/ProteusPass.cpp
  ${PROJECT_SOURCE_DIR}/src/runtime/Error.cpp
)

add_library(ProteusPassObjects OBJECT
  ${PROTEUS_PASS_SOURCES}
)

llvm_map_components_to_libnames(
  proteus_pass_offload_llvm_libs
  BitWriter
  IPO
  Linker
  Passes
)
set(proteus_pass_offload_link_libs)
foreach(lib IN LISTS proteus_pass_offload_llvm_libs)
  list(APPEND proteus_pass_offload_link_libs "$<LINK_ONLY:${lib}>")
endforeach()

set_target_properties(ProteusPassObjects PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(ProteusPassObjects
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(ProteusPassObjects
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ProteusPassObjects PRIVATE -fno-rtti)
endif()

add_library(ProteusPass SHARED
  $<TARGET_OBJECTS:ProteusPassObjects>
)

add_library(ProteusPassOffload SHARED
  $<TARGET_OBJECTS:ProteusPassObjects>
)

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(ProteusPass
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

target_link_libraries(ProteusPassOffload
  PRIVATE ${proteus_pass_offload_link_libs}
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
target_link_options(ProteusPassOffload
  PRIVATE "$<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,-Bsymbolic>")

target_compile_options(
  ProteusPass
  INTERFACE
  -fpass-plugin=$<TARGET_FILE:ProteusPass>
)

target_link_options(
  ProteusPass
  INTERFACE
  "SHELL:$<$<LINK_LANGUAGE:HIP>:-Xoffload-linker --load-pass-plugin=$<TARGET_FILE:ProteusPassOffload>>"
)

set_target_properties(ProteusPass ProteusPassOffload PROPERTIES
  # on macOS enable passing RPATH flags to the linker; no-ops on Linux
  MACOSX_RPATH          ON
  # let CMake use INSTALL_RPATH even in the build tree
  BUILD_WITH_INSTALL_RPATH  TRUE
  # pick @loader_path on Darwin, $ORIGIN elsewhere
  INSTALL_RPATH        "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,\$ORIGIN>"
)

install(
  TARGETS ProteusPass ProteusPassOffload
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
