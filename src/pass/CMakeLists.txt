add_library(ProteusPass SHARED ProteusPass.cpp )#DetectScops.cpp NonAffineSCEVs.cpp)

target_include_directories(ProteusPass
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(ProteusPass
  SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS}/polly)




if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(ProteusPass PRIVATE -fno-rtti)
endif()

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(ProteusPass
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
   stdc++fs)

target_link_options(ProteusPass PRIVATE "-Wl --version-script=${PROJECT_SOURCE_DIR}/version_script")
target_include_directories(ProteusPass PUBLIC  "/usr/tce/packages/clang/clang-ibm-18.1.8-cuda-11.8.0-gcc-11.2.1/include/polly")
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
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
