/**
 * @file
 */
#include "slab_node.h"
#include "traverse.h"
#include "device_allocator.h"
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda/std/utility>

#pragma once

namespace lslab {

template<int block_size, typename map_t, typename K, typename V>
__global__ void put_(map_t map, cuda::std::pair<K, V>* operations, V* output, size_t size);

template<int block_size, typename map_t, typename K, typename V>
__global__ void put_(map_t map, K* operations_keys, V* operations_values, V* output, size_t size);

template<int block_size, typename map_t, typename K, typename V>
__global__ void get_(map_t map, K* operations, cuda::std::pair<bool, V>* output, size_t size);

template<int block_size, typename map_t, typename K, typename V>
__global__ void update_(map_t map, cuda::std::pair<K, V>* operations, cuda::std::pair<bool, V>* output, size_t size);

template<typename T>
struct hash {
    LSLAB_HOST_DEVICE size_t operator()(T x) {
        return static_cast<size_t>(x);
    }
};

/**
 * LSlab map for GPU
 */
template<typename K, typename V, typename Allocator = device_allocator<slab_node<K, V>>, typename Hash = hash<K>>
class map {
public:

    using this_t = map<K, V, Allocator, Hash>;

    LSLAB_HOST map() : map(10) {
        
    }

    LSLAB_HOST map(unsigned n_log_2) : number_of_buckets_log_2(n_log_2) {
        size_t size = 1 << n_log_2;
        cudaMalloc(&lock_table, sizeof(warp_mutex) * size);

        cudaMemset(lock_table, 0, sizeof(warp_mutex) * size);

        cudaMalloc(&buckets, sizeof(slab_node<K, V>) * size);
        
        cudaMemset(buckets, 0, sizeof(slab_node<K, V>) * size);
    }

    LSLAB_HOST_DEVICE map(warp_mutex* lt, slab_node<K, V>* s, unsigned n_log_2) : lock_table(lt), buckets(s), number_of_buckets_log_2(n_log_2) {

    }

    LSLAB_HOST_DEVICE map(warp_mutex* lt, slab_node<K, V>* s, unsigned n_log_2, Allocator a) : lock_table(lt), buckets(s), number_of_buckets_log_2(n_log_2), alloc(a) {

    }

    LSLAB_HOST_DEVICE ~map() {
    } 

    template<typename Fn>
    LSLAB_DEVICE void find_function(const K& key, Fn&& fn, bool thread_mask = true) {

        size_t hash = Hash{}(key);
        hash &= ((1 << number_of_buckets_log_2) - 1);

        traverse<Allocator, OPERATION_TYPE::FIND>{}(lock_table, buckets, key, fn, alloc, hash, thread_mask); 
    }

    LSLAB_DEVICE bool get(const K& key, V& value, bool thread_mask = true) {
        struct Fn {
            LSLAB_DEVICE void operator()(const V& val) {
                value = val;
                found = true;
            }
            bool found;
            V& value;
        };

        Fn fn{false, value};

        find_function(key, fn, thread_mask);
        return fn.found;
    }

    template<typename Fn>
    LSLAB_DEVICE void insert_function(const K& key, Fn&& fn, bool thread_mask = true) {
        using traverse_t = traverse<Allocator, OPERATION_TYPE::INSERT>;
        traverse_t t;
        size_t hash = Hash{}(key) & ((1 << number_of_buckets_log_2) - 1);
        t.template operator()<K, V, Fn>(lock_table, buckets, key, std::forward<Fn>(fn), alloc, hash, thread_mask); 
    }

    LSLAB_DEVICE V put(const K& key, const V& value, bool thread_mask = true) {
        
        struct Fn_put {

            LSLAB_DEVICE void operator()(V& val) {
                tmp = val;
                val = value;
            }
            const V& value;
            V tmp;
        };

        Fn_put fn{value};

        insert_function(key, fn, thread_mask);
        return fn.tmp;
    }

    template<typename Fn>
    LSLAB_DEVICE bool update_function(const K& key, Fn&& fn, bool thread_mask = true) {
        traverse<Allocator, OPERATION_TYPE::UPDATE>{}(lock_table, buckets, key, fn, alloc, Hash{}(key) & ((1 << number_of_buckets_log_2) - 1), thread_mask); 
    }

    LSLAB_DEVICE cuda::std::pair<bool, V> update(const K& key, const V& value, bool thread_mask = true) {
        struct Fn {
            LSLAB_DEVICE void operator()(V& val) {
                tmp = value;
                val = value;
                found = true;
            }
            bool found;
            const V& value;
            V tmp;
        };

        Fn fn{false, value};

        update_function(key, fn, thread_mask);
        return {fn.found, fn.tmp};
    }

    template<int block_size = 256>
    LSLAB_HOST void put(cuda::std::pair<K, V>* operations, V* output, size_t size, cudaStream_t stream = 0x0) {
        put_<block_size, this_t, K, V><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, operations, output, size);
    }

    template<int block_size = 256>
    LSLAB_HOST void put(K* operations_keys, V* operations_values, V* output, size_t size, cudaStream_t stream = 0x0) {
        put_<block_size, this_t, K, V><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, operations_keys, operations_values, output, size);
    }


    template<int block_size = 256>
    LSLAB_HOST void get(K* operations, cuda::std::pair<bool, V>* output, size_t size, cudaStream_t stream = 0x0) {
        get_<block_size, this_t, K, V><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, operations, output, size);
    }

    template<int block_size = 256>
    LSLAB_HOST void update(cuda::std::pair<K, V>* operations, cuda::std::pair<bool, V>* output, size_t size, cudaStream_t stream = 0x0) {
        update_<block_size, this_t, K, V><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, operations, output, size);
    }

    LSLAB_HOST_DEVICE unsigned size() {
        return 1 << number_of_buckets_log_2;
    }

private:
    warp_mutex* lock_table;
    slab_node<K, V>* buckets;
    unsigned number_of_buckets_log_2;
    Allocator alloc;
};

template<int block_size, typename map_t, typename K, typename V>
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
