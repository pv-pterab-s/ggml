#include <stdlib.h>
#include <stdio.h>
#include <ggml-oneapi.h>
#include <CL/sycl.hpp>

extern "C" {

void ggml_init_oneapi(void) {
    printf("HERE\n");
}

void * ggml_oneapi_host_malloc(size_t size) {
    // OneAPI-specific code...
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_host(size, sycl::queue{});
    } catch (sycl::exception const& e) {
        std::cerr << "WARNING: failed to allocate " << size/1024.0/1024.0 << " MB of pinned memory: " << e.what() << "\n";
        return nullptr;
    }

    return ptr;
}

void ggml_oneapi_host_free(void * ptr) {
    // OneAPI-specific code...
    sycl::free(ptr, sycl::queue{});
}

void ggml_oneapi_memcpy(void * dst, const void * src, size_t size) {
    // OneAPI-specific code...
    // This would involve creating a SYCL queue and using the memcpy function.
    // You would need to handle the case where the source and destination are on different devices.
}

void ggml_oneapi_memset(void * ptr, int value, size_t size) {
    // OneAPI-specific code...
    // This would involve creating a SYCL queue and using the memset function.
}

}
