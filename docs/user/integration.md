# Integration

## CMake

Integration with CMake is straightforward.
Make sure the Proteus install directory is in `CMAKE_PREFIX_PATH`, or pass it
explicitly with `-Dproteus_DIR=<install_path>` during configuration.

Then, in your project's `CMakeLists.txt` add:
```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(<target>)
```

If you only need the DSL or C++ frontend APIs, you can link directly against
`proteusFrontend`.
In this case, you don’t need to compile your target with Clang:

```cmake
find_package(proteus CONFIG REQUIRED)

target_link_libraries(<target> ... proteusFrontend ...)
```

!!! warning
    You may need to build `libproteus` with the Position Independent Code (PIC)
    flag (`-fPIC` or `CMAKE_POSITION_INDEPENDENT_CODE=on` in CMake), if you want
    to integrate Proteus as a static library to a dynamic library target.

## Make

With `make`, integrating Proteus requires adding compilation and
linking flags, for example:

```bash
CXXFLAGS += -I<install_path>/include \
    -fpass-plugin=<install_path>/lib64/libProteusPass.so

LDFLAGS += -L <install_path>/lib64 \
    -Wl,-rpath,<install_path>/lib64 \
    -lproteus $(llvm-config --libs) -lclang-cpp
```

If you don't use code annotations, you can omit the `-fpass-plugin` option,
since he LLVM pass is only needed for processing annotations.

!!! note
    The example above links against `-lclang-cpp`, which is the monolithic
    Clang C++ library available in most LLVM distributions.
    If your toolchain instead provides static Clang component archives
    (e.g. `libclangFrontend.a`, `libclangDriver.a`, etc.), you’ll need to link
    those explicitly instead of `-lclang-cpp`. On ELF platforms, remember to wrap
    them in `-Wl,--start-group ... -Wl,--end-group` to resolve circular
    dependencies.

## Link with `-rdynamic`

When using Proteus for JIT generation of CPU code, and your JIT-compiled code
needs to call functions defined in your main program, you must link your main
application with the `-rdynamic` flag.
This flag ensures that symbols from your main program are exported and visible
to the Proteus JIT, allowing external function calls from JIT code to resolve
correctly at runtime.
