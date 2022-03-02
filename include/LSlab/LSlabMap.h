/**
 * @file
 */
#include "OperationsDevice.h"

#pragma once

namespace lslab {

/**
 * LSlab map for GPU
 */
template<typename K, typename V>
class LSlab {
public:

    LSLAB_HOST_DEVICE LSlab(volatile SlabData<K, V>** s, unsigned n, WarpAllocCtx<K, V> c) : slabs(s), number_of_buckets(n), ctx(c) {

    }

    LSLAB_HOST_DEVICE LSlab() : slabs(nullptr), number_of_buckets(0), ctx() {
    }

    LSLAB_HOST_DEVICE ~LSlab() {} 

    /// Warp get with mask
    LSLAB_DEVICE void get(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_search(threadMask, key, value, hash, slabs, number_of_buckets);
    }
    
    /// Warp put with mask
    LSLAB_DEVICE void put(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_replace(threadMask, key, value, hash, slabs, number_of_buckets, ctx);
    }
    
    /// Warp remove with mask
    LSLAB_DEVICE void remove(K& key, V& value, unsigned hash, bool threadMask = false) {
        warp_operation_delete(threadMask, key, value, hash, slabs, number_of_buckets);
    }

    /// Warp modify (update or remove) with mask
    LSLAB_DEVICE void modify(K& key, V& value, unsigned hash, Operation op, bool threadMask = false) {
        warp_operation_delete_or_replace(threadMask, key, value, hash, slabs, number_of_buckets, ctx, op);
    }

    LSLAB_HOST_DEVICE unsigned size() {
        return number_of_buckets;
    }

private:
    volatile SlabData<K, V> **slabs;
    unsigned number_of_buckets;
    WarpAllocCtx<K,V> ctx;
};

}
