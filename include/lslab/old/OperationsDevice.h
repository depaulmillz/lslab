/**
 * @file
 */
#include "LSlab.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/util_ptx.cuh>

#pragma once

namespace lslab {

/// Operations LSlab can perform
enum Operation {
    NOP = 0,
    GET = 2,
    PUT = 1,
    REMOVE = 3
};

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 32
#define VALID_KEY_MASK 0x7fffffffu
#define DELETED_KEY 0

const unsigned long long EMPTY_POINTER = 0;
#define BASE_SLAB 0

/// Since Slab
template<typename K, typename V>
struct SlabData {

    using KSub = typename std::conditional<sizeof(K) < sizeof(unsigned long long), unsigned long long, K>::type;

    static_assert(alignof(KSub) % alignof(void*) == 0, "Alignment must be a multiple of the pointer alignment");

    /// Used to lock with padding to 128B cacheline
    union {
        int ilock;
        alignas(128) char p[128];
    }; // 128 bytes

    /// Keys where key[31] is the pointer to next
    KSub key[32]; // 256 byte

    V value[32]; // 256 byte
    // the 32nd element is next
};

/// Memory block for warp allocation
template<typename K, typename V>
struct MemoryBlock {
    MemoryBlock() : bitmap(~0u), slab(nullptr) {
    }

    unsigned long long bitmap;
    /// 64 slabs
    SlabData<K, V> *slab;// 64 slabs
};

/// Struct for 32 MemoryBlocks
template<typename K, typename V>
struct SuperBlock {
    MemoryBlock<K, V> *memblocks;// 32 memblocks
};

/// Shfl wrapper
template<typename T>
LSLAB_DEVICE T shfl(unsigned mask, T val, int offset) {
    return cub::ShuffleIndex<32>(val, offset, mask);
}

/// Shfl wrapper for unsigned
template<>
LSLAB_DEVICE unsigned shfl(unsigned mask, unsigned val, int offset) {
    return __shfl_sync(mask, val, offset);
}

/// Shfl wrapper for int
template<>
LSLAB_DEVICE int shfl(unsigned mask, int val, int offset) {
    return __shfl_sync(mask, val, offset);
}

/// Shfl wrapper for ull
template<>
LSLAB_DEVICE unsigned long long shfl(unsigned mask, unsigned long long val, int offset) {
    return __shfl_sync(mask, val, offset);
}

/// Warp allocation contex
template<typename K, typename V>
struct WarpAllocCtx {
    WarpAllocCtx() : blocks(nullptr) {}

    /// Parallel shared nothing allocation of a slab
    LSLAB_DEVICE unsigned long long allocate() {
        // just doing parallel shared-nothing allocation
        const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
        const unsigned laneId = threadIdx.x & 0x1Fu;
        if (this->blocks == nullptr) {
            return 0;
        }
    
        MemoryBlock<K, V> *blocks = this->blocks[warpIdx].memblocks;
        unsigned bitmap = blocks[laneId].bitmap;
        int index = __ffs((int) bitmap) - 1;
        int ballotThread = __ffs((int) __ballot_sync(~0u, (index != -1))) - 1;
        if (ballotThread == -1) {
            if(laneId == 0)
                printf("Ran out of memory\n");
            __threadfence_system();
            __syncwarp();
            __trap();
            return 0;
        }

        auto location = (unsigned long long) (blocks[laneId].slab + index);
        if (ballotThread == laneId) {
            bitmap = bitmap ^ (1u << (unsigned) index);
            blocks[laneId].bitmap = bitmap;
        }
        location = shfl(~0u, location, ballotThread);
    
        return location;
    }
    
    /// Deallocation of a slab
    LSLAB_DEVICE void deallocate(unsigned long long l) {
    
        const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
        const unsigned laneId = threadIdx.x & 0x1Fu;
        if (this->blocks == nullptr) {
            return;
        }
    
        MemoryBlock<K, V> *blocks = this->blocks[warpIdx].memblocks;
        if ((unsigned long long) blocks[laneId].slab <= l && (unsigned long long) (blocks[laneId].slab + 32) > l) {
            unsigned diff = l - (unsigned long long) blocks[laneId].slab;
            unsigned idx = diff / sizeof(SlabData<K, V>);
            blocks[laneId].bitmap = blocks[laneId].bitmap | (1u << idx);
        }
    }

    SuperBlock<K, V> *blocks;
    // there should be a block per warp ie threadsPerBlock * blocks / 32 superblocks
};

/// SlabCtx is a wrapper for the slabs and number of buckets
template<typename K, typename V>
struct SlabCtx {
    SlabCtx() : slabs(nullptr), num_of_buckets(0) {}

    volatile SlabData<K, V> **slabs;
    unsigned num_of_buckets;
};

/**
 * There is a barrier after this locking
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
LSLAB_DEVICE void
LockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
         volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        while (atomicCAS(ilock, 0, -1) != 0);
    }
    __syncwarp();

}

/**
 * There is a barrier after this locking
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
LSLAB_DEVICE void
SharedLockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
               volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        while (true) {
            auto pred = *ilock;

            if (pred != -1 && atomicCAS(ilock, pred, pred + 1) == pred) {
                break;
            }
        }
    }
    __syncwarp();
}


/**
 * Note there is no barrier before or after, pay attention to reordering
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
LSLAB_DEVICE void
UnlockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
           volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *)  &(slabs[src_bucket]->ilock);
        atomicExch(ilock, 0);
    }

}

/**
 * Note there is no barrier before or after, pay attention to reordering
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
LSLAB_DEVICE void
SharedUnlockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
                 volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        atomicAdd(ilock, -1);
    }

}

/**
 * Reads the key at index laneId at the bucket next or src_bucket from slabs
 * @param next pointer to next either the pointer to a slab or null
 * @param src_bucket the bucket we are operating on
 * @param laneId the lane id of the thread
 * @param slabs the hashmap
 */
template<typename K, typename V>
LSLAB_DEVICE typename SlabData<K, V>::KSub
ReadSlabKey(const unsigned long long &next, const unsigned &src_bucket,
            const unsigned laneId, volatile SlabData<K, V> **slabs) {
    static_assert(sizeof(typename SlabData<K, V>::KSub) >= sizeof(void*), "Need to be able to substitute pointers for values");
    return next == BASE_SLAB ? const_cast<typename SlabData<K, V>::KSub&>(slabs[src_bucket]->key[laneId]) : 
        const_cast<typename SlabData<K, V>::KSub&>(reinterpret_cast<SlabData<K, V> *>(next)->key[laneId]);
}

/**
 * Reads the value at index laneId at the bucket next or src_bucket from slabs
 * @param next pointer to next either the pointer to a slab or null
 * @param src_bucket the bucket we are operating on
 * @param laneId the lane id of the thread
 * @param slabs the hashmap
 */
template<typename K, typename V>
LSLAB_DEVICE V
ReadSlabValue(const unsigned long long &next, const unsigned &src_bucket,
              const unsigned laneId, volatile SlabData<K, V> **slabs) {
    return (next == BASE_SLAB ? const_cast<SlabData<K, V>*>(slabs[src_bucket])->value[laneId] : reinterpret_cast<SlabData<K, V> *>(next)->value[laneId]);
}

/**
 * Adresses the key at index laneId at the bucket next or src_bucket from slabs
 * @param next pointer to next either the pointer to a slab or null
 * @param src_bucket the bucket we are operating on
 * @param laneId the lane id of the thread
 * @param slabs the hashmap
 * @param num_of_buckets
 */

template<typename K, typename V>
LSLAB_DEVICE volatile typename SlabData<K, V>::KSub *
SlabAddressKey(const unsigned long long &next, const unsigned &src_bucket,
               const unsigned laneId, volatile SlabData<K, V> **slabs,
               unsigned num_of_buckets) {
    return (volatile typename SlabData<K, V>::KSub *) ((next == BASE_SLAB ? slabs[src_bucket]->key : ((SlabData<K, V> *) next)->key) + laneId);
}

/**
 * Addresses the value at index laneId at the bucket next or src_bucket from slabs
 * @param next pointer to next either the pointer to a slab or null
 * @param src_bucket the bucket we are operating on
 * @param laneId the lane id of the thread
 * @param slabs the hashmap
 * @param num_of_buckets
 */
template<typename K, typename V>
LSLAB_DEVICE volatile V *
SlabAddressValue(const unsigned long long &next, const unsigned &src_bucket,
                 const unsigned laneId, volatile SlabData<K, V> **slabs,
                 unsigned num_of_buckets) {
    return (next == BASE_SLAB ? slabs[src_bucket]->value : ((SlabData<K, V> *) next)->value) + laneId;
}

/**
 * Searches the bucket modhash for the value associated with myKey
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash the bucket that has been hashed to
 * @param slabs
 * @param num_of_buckets
 */
template<typename K, typename V>
LSLAB_DEVICE void warp_operation_search(bool &is_active, const K &myKey,
                                                      V &myValue, const unsigned &modhash,
                                                      volatile SlabData<K, V> **__restrict__ slabs,
                                                      unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    const K threadKey = myKey;
    unsigned last_work_queue = 0;

    while (work_queue != 0) {

        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

        unsigned src_lane = __ffs(work_queue) - 1;
        K src_key = shfl(~0u, threadKey, src_lane);
        unsigned src_bucket = shfl(~0u, modhash, src_lane);
        if(work_queue != last_work_queue){
            SharedLockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
        auto read_key = ReadSlabKey(next, src_bucket, laneId, slabs);

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;

        if (masked_ballot != 0) {
            V read_value = ReadSlabValue(next, src_bucket, laneId, slabs);

            unsigned found_lane = __ffs(masked_ballot) - 1;
            auto found_value = shfl(~0u, read_value, found_lane);
            if (laneId == src_lane) {
                myValue = found_value;
                is_active = false;
            }
        } else {
            static_assert(sizeof(read_key) >= sizeof(void*), "Need read key to be bigger than the size of a pointer");
            unsigned long long next_ptr = shfl(~0u, reinterpret_cast<unsigned long long&>(read_key), ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            } else {
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;


        work_queue = __ballot_sync(~0u, is_active);

        if(work_queue != last_work_queue){
            SharedUnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
    }
}

/**
 * Returns value when removed or empty on removal
 * @tparam K
 * @tparam V
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash
 * @param slabs
 * @param num_of_buckets
 */
template<typename K, typename V>
LSLAB_DEVICE void
warp_operation_delete(bool &is_active, const K &myKey,
                      V &myValue, const unsigned &modhash,
                      volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);
    unsigned last_work_queue = 0;

    while (work_queue != 0) {
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
        unsigned src_lane = __ffs(work_queue) - 1;
        auto src_key = shfl(~0u, myKey, src_lane);
        unsigned src_bucket = shfl(~0u, modhash, src_lane);
        
        if(work_queue != last_work_queue){
            LockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }

        K read_key = ReadSlabKey(next, src_bucket, laneId, slabs);

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;

        if (masked_ballot != 0) {

            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                *const_cast<K*>(SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets)) = K{};
                is_active = false;
                myValue = ReadSlabValue(next, src_bucket, dest_lane, slabs);
                //success = true;
                __threadfence();
            }
        } else {
            static_assert(sizeof(read_key) >= sizeof(void*), "Need read key to be bigger than the size of a pointer");
            unsigned long long next_ptr = shfl(~0u, reinterpret_cast<unsigned long long&>(read_key), ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                is_active = false;
                myValue = V{};
                //success = false;
            } else {
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
        if(work_queue != last_work_queue){
            UnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
    }
}

/// Replaces the value of myKey with myValue in bucket modhash
template<typename K, typename V>
LSLAB_DEVICE void
warp_operation_replace(bool &is_active, const K &myKey,
                       V &myValue, const unsigned &modhash,
                       volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = 0;

    bool foundEmptyNext = false;
    unsigned long long empty_next = BASE_SLAB;

    while (work_queue != 0) {
        
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

        auto src_lane = __ffs( work_queue) - 1;
        auto src_key = shfl(~0u, myKey, src_lane);
        unsigned src_bucket = shfl(~0u, modhash, src_lane);

        if(work_queue != last_work_queue){
            foundEmptyNext = false;
            LockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }

        K read_key = ReadSlabKey(next, src_bucket, laneId, slabs);

        bool to_share = read_key == src_key;
        auto masked_ballot = __ballot_sync(~0u, to_share) & VALID_KEY_MASK;

        if(!foundEmptyNext && read_key == K{}){
            foundEmptyNext = true;
            empty_next = next;
        }

        if (masked_ballot != 0) {
            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                volatile K *addrKey = SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets);
                volatile V *addrValue = SlabAddressValue(next, src_bucket, dest_lane, slabs, num_of_buckets);
                V tmpValue = V{};
                if (*const_cast<K*>(addrKey) == K{}) {
                    *const_cast<K*>(addrKey) = myKey;
                } else {
                    tmpValue = *const_cast<V*>(addrValue);
                }
                *const_cast<V*>(addrValue) = myValue;
                myValue = tmpValue;
                __threadfence_system();
                is_active = false;
            }
        } else {
            static_assert(sizeof(read_key) >= sizeof(void*), "Need read key to be bigger than the size of a pointer");
            unsigned long long next_ptr = shfl(~0u, reinterpret_cast<unsigned long long&>(read_key), ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                __threadfence_system();
                masked_ballot = (int) (__ballot_sync(~0u, foundEmptyNext) & VALID_KEY_MASK);
                if (masked_ballot != 0) {
                    unsigned dest_lane = __ffs(masked_ballot) - 1;
                    unsigned long long new_empty_next = shfl(~0u, empty_next, dest_lane);
                    if (src_lane == laneId) {
                        volatile K *addrKey = SlabAddressKey(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                        volatile V *addrValue = SlabAddressValue(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                        V tmpValue = V{};
                        if (*const_cast<K*>(addrKey) == K{}) {
                            *const_cast<K*>(addrKey) = src_key;
                        } else {
                            tmpValue = *const_cast<V*>(addrValue);
                        }
                        *const_cast<V*>(addrValue) = myValue;
                        myValue = tmpValue;
                        __threadfence_system();
                        is_active = false;
                    }
                } else {
                    unsigned long long new_slab_ptr = ctx.allocate();
                    if (laneId == ADDRESS_LANE - 1) {
                        volatile K *slabAddr = SlabAddressKey(next, src_bucket, ADDRESS_LANE - 1,
                                                                               slabs, num_of_buckets);
                        *reinterpret_cast<volatile unsigned long long*>(slabAddr) = new_slab_ptr;
                        __threadfence_system();
                    }
                    next = new_slab_ptr;
                }
            } else {
                next = next_ptr;
            }
        }
        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);

        if (work_queue != last_work_queue) {
            UnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
    }
}

/**
 * Returns value when removed or empty on removal
 * @tparam K
 * @tparam V
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash
 * @param slabs
 * @param num_of_buckets
 */
template<typename K, typename V>
LSLAB_DEVICE void
warp_operation_delete_or_replace(bool &is_active, const K &myKey,
                      V &myValue, const unsigned &modhash,
                      volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx, Operation op) {

    using KSub = typename SlabData<K, V>::KSub;
    
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);
    unsigned last_work_queue = 0;
    bool foundEmptyNext = false;
    unsigned long long empty_next = BASE_SLAB;

    while (work_queue != 0) {
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
        auto src_lane = __ffs(work_queue) - 1;
        auto src_key = shfl(~0u, myKey, src_lane);
        unsigned src_bucket = shfl(~0u, modhash, (int) src_lane);
        
        if(work_queue != last_work_queue){
            foundEmptyNext = false;
            LockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }

        KSub read_key = ReadSlabKey(next, src_bucket, laneId, slabs);

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;
        
        if(!foundEmptyNext && read_key == K{}){
            foundEmptyNext = true;
            empty_next = next;
        }
        
        if (masked_ballot != 0) {

            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;

                if(op == REMOVE) {
                    *const_cast<KSub *>(SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets)) = K{};
                    is_active = false;
                    myValue = ReadSlabValue(next, src_bucket, dest_lane, slabs);
                    //success = true;
                } else {
                    volatile KSub *addrKey = SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets);
                    auto *addrValue = SlabAddressValue(next, src_bucket, dest_lane, slabs, num_of_buckets);
                    V tmpValue = V{};
                    K addrKeyDeref = const_cast<KSub&>(*addrKey);
                    if (addrKeyDeref == K{}) {
                        *const_cast<KSub*>(addrKey) = myKey;
                    } else {
                        tmpValue = *const_cast<V*>(addrValue);
                    }
                    *const_cast<V*>(addrValue) = myValue;
                    myValue = tmpValue;
                    is_active = false;
                }
                __threadfence_system();
            }

        } else {
            static_assert(sizeof(read_key) >= sizeof(void*), "Need read key to be bigger than the size of a pointer");
            unsigned long long next_ptr = shfl(~0u, reinterpret_cast<unsigned long long&>(read_key), ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                if(op == REMOVE) {
                    is_active = false;
                    myValue = V{};
                } else {
                    __threadfence_system();
                    masked_ballot = (int) (__ballot_sync(~0u, foundEmptyNext) & VALID_KEY_MASK);
                    if (masked_ballot != 0) {
                        unsigned dest_lane = __ffs(masked_ballot) - 1;
                        unsigned new_empty_next = shfl(~0u, empty_next, dest_lane);
                        if (src_lane == laneId) {
                            auto addrKey = SlabAddressKey(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                            auto addrValue = SlabAddressValue(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                            V tmpValue = V{};
                            K addrKeyDeref = const_cast<KSub&>(*addrKey);
                            if (addrKeyDeref == K{}) {
                                *const_cast<KSub*>(addrKey) = src_key;
                            } else {
                                tmpValue = *const_cast<V*>(addrValue);
                            }
                            *const_cast<V*>(addrValue) = myValue;
                            myValue = tmpValue;
                            __threadfence_system();
                            is_active = false;
                        }
                    } else {
                        unsigned long long new_slab_ptr = ctx.allocate();
                        if (laneId == ADDRESS_LANE - 1) {
                            auto *slabAddr = SlabAddressKey(next, src_bucket, ADDRESS_LANE - 1, slabs, num_of_buckets);
                            *reinterpret_cast<volatile unsigned long long*>(slabAddr) = new_slab_ptr;
                            __threadfence_system();
                        }
                        next = new_slab_ptr;
                    }
                }
                //success = false;
            } else {
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
        if(work_queue != last_work_queue){
            UnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
    }
}

}
