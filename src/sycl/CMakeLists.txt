set(SYCL_SOURCES
    # Add your SYCL implementation source files here
    # We're handling the main SYCL implementations in the util directory
    # but additional SYCL-specific files can be added here
)


# SYCL-specific compiler flags and options
if(BATCHLAS_ENABLE_SYCL AND TARGET batchlas)
    # Intel oneAPI DPC++ specific settings
    target_compile_options(batchlas PRIVATE -fsycl)
    target_link_options(batchlas PRIVATE -fsycl)
    
    # If needed, you can add specific device targets here
    # For example, to target Intel GPUs:
    # target_compile_options(batchlas PRIVATE -fsycl-targets=spir64_gen)
endif()

target_sources(batchlas_sycl_obj PRIVATE ${SYCL_SOURCES})