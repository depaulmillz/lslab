/**
 * @file
 */

#include "lslab.h"
#include "mutex.h"
#include <stdexcept>

#pragma once

namespace lslab {

template<typename T>
struct device_allocator {

    LSLAB_HOST device_allocator(size_t stack_size_ = 10000); 
    
    LSLAB_HOST_DEVICE device_allocator(const device_allocator<T>& self);
    
    LSLAB_HOST_DEVICE device_allocator(device_allocator&& other); 

    LSLAB_HOST_DEVICE device_allocator& operator=(device_allocator&& other);
  
    LSLAB_HOST_DEVICE ~device_allocator();    
    
    LSLAB_DEVICE T* allocate(size_t n);

    LSLAB_HOST_DEVICE void deallocate(T* ptr, size_t n); 

    size_t stack_size;
    cuda::std::atomic<uint64_t>* loc;
    T* mempool;

};

}

#include "detail/device_allocator.h"
