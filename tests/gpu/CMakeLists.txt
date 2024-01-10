if(ENABLE_HIP)
    enable_language(HIP)
elseif(ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to compile for")
    endif()

    enable_language(CUDA)
    message(STATUS "CUDA compiler ${CMAKE_CUDA_COMPILER_ID}")

    if(NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
        message(FATAL_ERROR "JIT is compatible only with Clang CUDA compilation")
    endif()
endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py "
import lit.formats

config.name = 'LIT tests'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.cpp']
config.test_source_root = '${CMAKE_CURRENT_SOURCE_DIR}'
config.test_exec_root = '${CMAKE_CURRENT_BINARY_DIR}'
ext = lit_config.params['EXT']
config.substitutions.append(('%ext', ext))
"
)

CREATE_GPU_TEST(kernel kernel.cpp)
CREATE_GPU_TEST(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST(kernel_args kernel_args.cpp)
CREATE_GPU_TEST(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST(multi_file file1_kernel.cpp file2_kernel.cpp)
CREATE_GPU_TEST(daxpy daxpy.cpp)
CREATE_GPU_TEST(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST(kernel_host_device_jit kernel_host_device_jit.cpp)

CREATE_GPU_TEST_RDC(kernel kernel.cpp)
CREATE_GPU_TEST_RDC(kernel_cache kernel_cache.cpp)
CREATE_GPU_TEST_RDC(kernel_args kernel_args.cpp)
CREATE_GPU_TEST_RDC(kernels_gvar kernels_gvar.cpp)
CREATE_GPU_TEST_RDC(kernel_launches kernel_launches.cpp)
CREATE_GPU_TEST_RDC(kernel_launches_args kernel_launches_args.cpp)
CREATE_GPU_TEST_RDC(indirect_launcher indirect_launcher.cpp)
CREATE_GPU_TEST_RDC(daxpy daxpy.cpp)
CREATE_GPU_TEST_RDC(kernel_host_jit kernel_host_jit.cpp)
CREATE_GPU_TEST_RDC(kernel_host_device_jit kernel_host_device_jit.cpp)
