#include "lslab.h"
#include "mutex.h"

#pragma once

namespace lslab {

template<typename T>
struct device_allocator {

    LSLAB_HOST device_allocator(size_t stack_size_ = 10000) : stack_size(stack_size_) {
        gpuErrchk(cudaMalloc(&loc, sizeof(cuda::std::atomic<uint64_t>))); 
        gpuErrchk(cudaMemset(loc, 0, sizeof(cuda::std::atomic<uint64_t>))); 
        gpuErrchk(cudaMalloc(&mempool, sizeof(T) * stack_size)); 
        gpuErrchk(cudaMemset(mempool, 0, sizeof(T) * stack_size)); 
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
