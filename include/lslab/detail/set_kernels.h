#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda/std/utility>

#pragma once

namespace lslab {
namespace set_kernels {

template<int block_size, typename map_t, typename K>
__launch_bounds__(block_size)
__global__ void remove_(map_t map, const K* keys, bool* output, size_t size) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    K key; 
    if(tidx + bidx * block_size < size) {
        key = keys[tidx + bidx * block_size];
    }

    bool res = map.remove(key, tidx + bidx * block_size < size);

    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }
}

template<int block_size, typename map_t, typename K>
__launch_bounds__(block_size)
__global__ void contains_(map_t map, const K* keys, bool* output, size_t size) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    K key; 
    if(tidx + bidx * block_size < size) {
        key = keys[tidx + bidx * block_size];
    }

    bool res = map.contains(key, tidx + bidx * block_size < size);

    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }
}


template<int block_size, typename map_t, typename K>
__launch_bounds__(block_size)
__global__ void insert_(map_t map, const K* keys, bool* output, size_t size) {

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    
    K key;
    if(tidx + bidx * block_size < size) {
        key = keys[tidx + bidx * block_size];
    }

    bool res = map.insert(key, tidx + bidx * block_size < size);
    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }

}

}
}
