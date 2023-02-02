/**
 * @file
 */
#include <cuda.h>
#include <cuda/std/utility>

#pragma once

namespace lslab {
namespace map_kernels {

template<int block_size, typename map_t, typename K, typename V>
__launch_bounds__(block_size)
__global__ void put_(map_t map, cuda::std::pair<K, V>* operations, V* output, size_t size) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    K key; 
    V val;
    if(tidx + bidx * block_size < size) {
        key = operations[tidx + bidx * block_size].first;
        val = operations[tidx + bidx * block_size].second;
    }

    V res = map.put(key, val, tidx + bidx * block_size < size);

    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }
}

template<int block_size, typename map_t, typename K, typename V>
__launch_bounds__(block_size)
__global__ void put_(map_t map, K* operations_keys, V* operations_values, V* output, size_t size) {
    
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    K key; 
    V val;
    if(tidx + bidx * block_size < size) {
        key = operations_keys[tidx + bidx * block_size];
        val = operations_values[tidx + bidx * block_size];
    }

    V res = map.put(key, val, tidx + bidx * block_size < size);

    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }
}


template<int block_size, typename map_t, typename K, typename V>
__launch_bounds__(block_size)
__global__ void get_(map_t map, K* operations, cuda::std::pair<bool, V>* output, size_t size) {

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    
    K key;
    V value;
    if(tidx + bidx * block_size < size) {
        key = operations[tidx + bidx * block_size];
    }

    bool res = map.get(key, value, tidx + bidx * block_size < size);
    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = {res, value};
    }

}

template<int block_size, typename map_t, typename K, typename V>
__launch_bounds__(block_size)
__global__ void update_(map_t map, cuda::std::pair<K, V>* operations, cuda::std::pair<bool, V>* output, size_t size) {

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    K key; 
    V val;
    if(tidx + bidx * block_size < size) {
        key = operations[tidx + bidx * block_size].first;
        val = operations[tidx + bidx * block_size].second;
    }

    cuda::std::pair<bool, V> res = map.put(key, val, tidx + bidx * block_size < size);
    if(tidx + bidx * block_size < size) {
        output[tidx + bidx * block_size] = res;
    }
}

}
}
