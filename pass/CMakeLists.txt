add_library(ProteusPass SHARED ProteusJitPass.cpp)

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

install(
  TARGETS ProteusPass
  EXPORT proteusTargets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
