add_subdirectory(cpu)

if(PROTEUS_ENABLE_CUDA OR PROTEUS_ENABLE_HIP)
    add_subdirectory(gpu)
    add_subdirectory(host_device)
endif()
