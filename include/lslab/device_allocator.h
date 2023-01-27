#include "lslab.h"
#include "mutex.h"
#include <stdexcept>

#pragma once

namespace lslab {

template<typename T>
struct device_allocator {

    LSLAB_HOST device_allocator(size_t stack_size_ = 10000) : stack_size(stack_size_) {
        gpuErrchk(cudaMalloc(&loc, sizeof(cuda::std::atomic<uint64_t>))); 
        gpuErrchk(cudaMemset(loc, 0, sizeof(cuda::std::atomic<uint64_t>))); 
        gpuErrchk(cudaMalloc(&mempool, sizeof(T) * stack_size)); 
    }

    LSLAB_HOST_DEVICE device_allocator(const device_allocator<T>& self) : stack_size(self.stack_size), loc(self.loc), mempool(self.mempool) {}
    
    LSLAB_HOST_DEVICE device_allocator(device_allocator&& other) {
        stack_size = other.stack_size;
        loc = other.loc;
        mempool = other.mempool;

        other.loc = nullptr;
        other.mempool = nullptr;
    }

    LSLAB_HOST_DEVICE device_allocator& operator=(device_allocator&& other) {
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

    LSLAB_HOST_DEVICE ~device_allocator() {

        // we let it leak for now

        //#if !defined(__CUDA_ARCH__)
        //    if(loc != nullptr)
        //        gpuErrchk(cudaFree(static_cast<void*>(loc)));
        //    if(mempool != nullptr)
        //        gpuErrchk(cudaFree(mempool));
        //#endif
    }
    
    LSLAB_DEVICE T* allocate(size_t n) {
        auto idx = loc->fetch_add(n);
        if(idx >= stack_size) {
            printf("At idx %llu\n", idx);
            __trap();
        }
        return mempool + idx;
    }

    LSLAB_HOST_DEVICE void deallocate(T* ptr, size_t n) {}

    size_t stack_size;
    cuda::std::atomic<uint64_t>* loc;
    T* mempool;

};

};
