# Integration

To integrate Proteus into your application, you must:

1. Annotate functions (or GPU kernels) for JIT specialization.
2. Modify your build to include: (a) the Proteus LLVM plugin pass, (b) the
include directory with Proteus's headers, and link with the runtime library.

This is done by adding Proteus's plugin pass to Clang compilation and its
include directory:
```shell
CXXFLAGS += -fpass-plugin=<install-path>/lib64/libProteusPass.so -I<install-path>/include
```

Also, linking must include the runtime library (preferrably rpath-ed is using
Proteus as a dynamic library) and LLVM libraries:
```shell
LDFLAGS += -L <install-path>/lib64 -Wl,-rpath,<install-path>/lib -lproteus $(llvm-config --libs)
```

A complete example is:
```shell
clang++ -fpass-plugin=<install-path>/lib64/libProteusPass.so \
    -I<install-path>/include \
    -L <install-path>/lib64 -Wl,-rpath,<install-path>/lib64 \
    -lproteus $(llvm-config --libs) \
    MyAwesomeCode.cpp -o MyAwesomeExe
```

## Using CMake

To use Proteus with CMake, make sure the Proteus install directory is in
`CMAKE_PREFIX_PATH`, or pass it as `-Dproteus_DIR=<install-path>`.
Then, in your project's `CMakeLists.txt` simply add the following two lines:

```cmake
find_package(proteus CONFIG REQUIRED)

add_proteus(target)
```

Where `target` is the name of your library or executable target.

!!! warning
    You may need to build `libproteus` with the Position Independent Code (PIC)
    flag (`-fPIC` or `CMAKE_POSITION_INDEPENDENT_CODE=on` in CMake), if you want
    to integrate Proteus as a static library to a dynamic library target.
