add_library(IRLinker SHARED IRLinker.cpp)
target_include_directories(IRLinker
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(IRLinker
  SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

if(APPLE)
  # On macOS we avoid the use of the -Bsymbolic linker flag.
  target_link_libraries(IRLinker
  PRIVATE LLVMLinker proteus)
else()
  target_link_libraries(IRLinker
  PRIVATE LLVMLinker
  proteus
  -Wl,-Bsymbolic)
endif()

# We use the full path because we query those source files to make them
# dependencies for building tests.
set(PROTEUS_PASS_SOURCES
  ${PROJECT_SOURCE_DIR}/src/pass/AnnotationHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/pass/ProteusPass.cpp
  ${PROJECT_SOURCE_DIR}/src/lib/Error.cpp
)

add_library(ProteusPass SHARED
  ${PROTEUS_PASS_SOURCES}
)

target_include_directories(ProteusPass
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(ProteusPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ProteusPass PRIVATE -fno-rtti)
endif()

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(ProteusPass
  PRIVATE IRLinker
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

set_target_properties(ProteusPass PROPERTIES
  # on macOS enable passing RPATH flags to the linker; no-ops on Linux
  MACOSX_RPATH          ON
  # let CMake use INSTALL_RPATH even in the build tree
  BUILD_WITH_INSTALL_RPATH  TRUE
  # pick @loader_path on Darwin, $ORIGIN elsewhere
  INSTALL_RPATH        "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,\$ORIGIN>"
)

install(
  TARGETS ProteusPass IRLinker
  EXPORT proteusTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
