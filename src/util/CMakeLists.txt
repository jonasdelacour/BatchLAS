set(UTIL_SOURCES
    queue-impl.cc
    sycl-util-impl.cc
)

target_sources(batchlas PRIVATE ${UTIL_SOURCES})

# Add specific SYCL compiler flags for these files
#set_source_files_properties(
#    queue-impl.cc
#    sycl-util-impl.cc
#    PROPERTIES COMPILE_FLAGS "-fsycl"
#)