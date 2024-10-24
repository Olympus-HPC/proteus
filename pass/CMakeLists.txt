add_library(ProteusJitPass SHARED ProteusJitPass.cpp)

target_include_directories(ProteusJitPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ProteusJitPass PRIVATE -fno-rtti)
endif()

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(ProteusJitPass
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

install(TARGETS ProteusJitPass)
