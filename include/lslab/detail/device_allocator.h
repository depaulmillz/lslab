#include "../lslab.h"
#include "../mutex.h"
#include <stdexcept>

#pragma once

namespace lslab {

template<typename T>
LSLAB_HOST device_allocator<T>::device_allocator(size_t stack_size_) : stack_size(stack_size_) {
    gpuErrchk(cudaMalloc(&loc, sizeof(cuda::std::atomic<uint64_t>))); 
    gpuErrchk(cudaMemset(loc, 0, sizeof(cuda::std::atomic<uint64_t>))); 
    gpuErrchk(cudaMalloc(&mempool, sizeof(T) * stack_size)); 
}

template<typename T>
LSLAB_HOST_DEVICE device_allocator<T>::device_allocator(const device_allocator<T>& self) : stack_size(self.stack_size), loc(self.loc), mempool(self.mempool) {}

template<typename T>
LSLAB_HOST_DEVICE device_allocator<T>::device_allocator(device_allocator&& other) {
    stack_size = other.stack_size;
    loc = other.loc;
    mempool = other.mempool;

    other.loc = nullptr;
    other.mempool = nullptr;
}

template<typename T>
LSLAB_HOST_DEVICE device_allocator<T>& device_allocator<T>::operator=(device_allocator<T>&& other) {
    #if defined(__CUDA_ARCH__)
        if(loc != nullptr || mempool != nullptr)
           __trap();
    #else
        if(loc != nullptr)
            gpuErrchk(cudaFree(loc));
        if(mempool != nullptr)
            gpuErrchk(cudaFree(mempool));
    #endif

    stack_size = other.stack_size;
    loc = other.loc;
    mempool = other.mempool;

    other.loc = nullptr;
    other.mempool = nullptr;
}

template<typename T>
LSLAB_HOST_DEVICE device_allocator<T>::~device_allocator() {

    // we let it leak for now

    //#if !defined(__CUDA_ARCH__)
    //    if(loc != nullptr)
    //        gpuErrchk(cudaFree(static_cast<void*>(loc)));
    //    if(mempool != nullptr)
    //        gpuErrchk(cudaFree(mempool));
    //#endif
}

template<typename T>
LSLAB_DEVICE T* device_allocator<T>::allocate(size_t n) {
    auto idx = loc->fetch_add(n);
    if(idx >= stack_size) {
        printf("At idx %llu\n", idx);
        __trap();
    }
    return mempool + idx;
}

template<typename T>
LSLAB_HOST_DEVICE void device_allocator<T>::deallocate(T* ptr, size_t n) {}

}
