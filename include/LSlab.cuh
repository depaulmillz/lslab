#include "Operations.cuh"

#pragma once

namespace lslab {

template<typename K, typename V>
class LSlab {
public:

    __forceinline__ __host__ __device__ LSlab(volatile SlabData<K, V>** s, unsigned n, WarpAllocCtx<K, V> c) : slabs(s), number_of_buckets(n), ctx(c) {

    }

    __forceinline__ __host__ __device__ ~LSlab() {} 

    __forceinline__ __host__ __device__ void get(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_search(threadMask, key, value, hash, slabs, number_of_buckets);
    }
    
    __forceinline__ __host__ __device__ void put(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_replace(threadMask, key, value, hash, slabs, number_of_buckets, ctx);
    }
    
    __forceinline__ __host__ __device__ void remove(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_delete(threadMask, key, value, hash, slabs, number_of_buckets);
    }

private:
    volatile SlabData<K, V> **slabs;
    unsigned number_of_buckets;
    WarpAllocCtx<K,V> ctx;
};

}
