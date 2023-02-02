#include "../lslab.h"
#include "../warp_mutex.h"
#include "slab_node.h"
#include <cuda.h>
#include <cub/util_ptx.cuh>

#pragma once

namespace lslab {

namespace detail {

enum OPERATION_TYPE {
    INSERT,
    UPDATE,
    FIND,
    REMOVE
};

template<typename Allocator,
         int TYPE>
struct traverse {
    
    template<typename T>
    using Lock_t = typename std::conditional<TYPE == OPERATION_TYPE::FIND, warp_shared_lock<T>, warp_unique_lock<T>>::type;

    template<typename K, typename V, typename Fn>
    LSLAB_DEVICE void operator()(warp_mutex* lock_table,
                                 slab_node<K, V>* buckets, 
                                 const K& key, 
                                 Fn&& fn,
                                 Allocator& alloc, 
                                 const unsigned bucket, 
                                 bool thread_mask) {

        const unsigned laneId = threadIdx.x % 32;
        slab_node<K, V>* next = nullptr;

        unsigned work_queue = __ballot_sync(~0u, thread_mask);

        unsigned last_work_queue = 0;

        Lock_t<warp_mutex> lock;

        while (work_queue != 0) {

            next = (work_queue != last_work_queue) ? nullptr : next;
            unsigned src_lane = __ffs(work_queue) - 1;
            
            K src_key = cub::ShuffleIndex<32>(key, src_lane, ~0);
            
            unsigned src_bucket = __shfl_sync(~0u, bucket, src_lane);

            bool found_empty = false;
            slab_node<K, V>* empty_location = nullptr;
           
            if(work_queue != last_work_queue) {
                next = &buckets[src_bucket];
                lock = Lock_t<warp_mutex>(lock_table[src_bucket]);
                //__threadfence_system();
            }

            slab_node<K, V>* next_ptr = nullptr;
            K read_key;

            bool found = false;

            
            if(laneId < 31) {
                read_key = next->key[laneId];
                unsigned valid = next->valid;
                valid = (valid >> laneId) & 0x1;
                if(valid) {
                    found = read_key == src_key;
                } else if(TYPE == OPERATION_TYPE::INSERT && !found_empty) {
                    found_empty = true;
                    empty_location = next; 
                }
            } else {
                next_ptr = next->next;
            }

            auto masked_ballot = __ballot_sync(~0u, found); //& 0x7fffffffu;

            if (masked_ballot != 0) {
                unsigned found_lane = __ffs(masked_ballot) - 1;
                if(laneId == src_lane) {
                    if(TYPE == OPERATION_TYPE::REMOVE) {
                        next->valid ^= (1 << found_lane);
                    }
                    fn(next->value[found_lane]);
                    thread_mask = false;
                }
            } else {
                next_ptr = reinterpret_cast<slab_node<K, V>*>(__shfl_sync(~0, reinterpret_cast<unsigned long long>(next_ptr), 31));
                if (next_ptr == nullptr) {

                    if(TYPE == OPERATION_TYPE::INSERT) {
                        // if we are doing an insert here
                        // check if found empty and then insert there
                        // otherwise allocate
                        masked_ballot = __ballot_sync(~0u, found_empty);
                        if(masked_ballot != 0) {
                            unsigned found_lane = __ffs(masked_ballot) - 1;
                            auto loc = reinterpret_cast<unsigned long long>(empty_location);
                            next = reinterpret_cast<slab_node<K, V>*>(__shfl_sync(~0, loc, found_lane));
                            if(laneId == src_lane) {
                                next->key[found_lane] = key;
                                next->valid |= (1 << found_lane);
                                fn(next->value[found_lane]);
                                // mark it valid
                                thread_mask = false;
                            }
                        } else {
                            if(laneId == 31) {
                                next_ptr = alloc.allocate(1);
                                next_ptr = new (static_cast<void*>(next_ptr)) slab_node<K, V>();
                                next->next = next_ptr;
                                next = next_ptr;
                            }
                            next = reinterpret_cast<slab_node<K, V>*>(__shfl_sync(~0, reinterpret_cast<unsigned long long>(next), 31));
                            if(next == nullptr)
                                __trap();
                        }
                    }

                    if ((TYPE == OPERATION_TYPE::FIND || TYPE == OPERATION_TYPE::UPDATE || TYPE == OPERATION_TYPE::REMOVE) && laneId == src_lane) {
                        // on read only did not find
                        thread_mask = false;
                    }
                } else {
                    next = next_ptr;
                }
            }

            last_work_queue = work_queue;

            work_queue = __ballot_sync(~0u, thread_mask);

            if(work_queue != last_work_queue){
                //__threadfence_system();
                lock = Lock_t<warp_mutex>();
            }
        }
    }

    template<typename K>
    LSLAB_DEVICE void operator()(warp_mutex* lock_table,
                                 set_node<K>* buckets, 
                                 const K& key,
                                 bool& result, 
                                 Allocator& alloc, 
                                 const unsigned bucket, 
                                 bool thread_mask) {

        const unsigned laneId = threadIdx.x % 32;
        set_node<K>* next = nullptr;

        unsigned work_queue = __ballot_sync(~0u, thread_mask);

        unsigned last_work_queue = 0;

        Lock_t<warp_mutex> lock;

        while (work_queue != 0) {

            next = (work_queue != last_work_queue) ? nullptr : next;
            unsigned src_lane = __ffs(work_queue) - 1;
            
            K src_key = cub::ShuffleIndex<32>(key, src_lane, ~0);
            
            unsigned src_bucket = __shfl_sync(~0u, bucket, src_lane);

            bool found_empty = false;
            set_node<K>* empty_location = nullptr;
           
            if(work_queue != last_work_queue) {
                next = &buckets[src_bucket];
                lock = Lock_t<warp_mutex>(lock_table[src_bucket]);
            }

            set_node<K>* next_ptr = nullptr;
            K read_key;

            bool found = false;

            
            if(laneId < 31) {
                read_key = next->key[laneId];
                unsigned valid = next->valid;
                valid = (valid >> laneId) & 0x1;
                if(valid) {
                    found = read_key == src_key;
                } else if(TYPE == OPERATION_TYPE::INSERT && !found_empty) {
                    found_empty = true;
                    empty_location = next; 
                }
            } else {
                next_ptr = next->next;
            }

            auto masked_ballot = __ballot_sync(~0u, found); //& 0x7fffffffu;

            if (masked_ballot != 0) {
                unsigned found_lane = __ffs(masked_ballot) - 1;
                if(laneId == src_lane) {
                    if(TYPE == OPERATION_TYPE::REMOVE) {
                        next->valid ^= (1 << found_lane);
                    }
                    result = true;
                    thread_mask = false;
                }
            } else {
                next_ptr = reinterpret_cast<set_node<K>*>(__shfl_sync(~0, reinterpret_cast<unsigned long long>(next_ptr), 31));
                if (next_ptr == nullptr) {

                    if(TYPE == OPERATION_TYPE::INSERT) {
                        // if we are doing an insert here
                        // check if found empty and then insert there
                        // otherwise allocate
                        masked_ballot = __ballot_sync(~0u, found_empty);
                        if(masked_ballot != 0) {
                            unsigned found_lane = __ffs(masked_ballot) - 1;
                            auto loc = reinterpret_cast<unsigned long long>(empty_location);
                            next = reinterpret_cast<set_node<K>*>(__shfl_sync(~0, loc, found_lane));
                            if(laneId == src_lane) {
                                next->key[found_lane] = key;
                                next->valid |= (1 << found_lane);
                                // mark it valid
                                thread_mask = false;
                                result = true;
                            }
                        } else {
                            if(laneId == 31) {
                                next_ptr = alloc.allocate(1);
                                next_ptr = new (static_cast<void*>(next_ptr)) set_node<K>();
                                next->next = next_ptr;
                                next = next_ptr;
                            }
                            next = reinterpret_cast<set_node<K>*>(__shfl_sync(~0, reinterpret_cast<unsigned long long>(next), 31));
                            if(next == nullptr)
                                __trap();
                        }
                    }

                    if ((TYPE == OPERATION_TYPE::FIND || TYPE == OPERATION_TYPE::REMOVE) && laneId == src_lane) {
                        // on read only did not find
                        thread_mask = false;
                        result = false;
                    }
                } else {
                    next = next_ptr;
                }
            }

            last_work_queue = work_queue;

            work_queue = __ballot_sync(~0u, thread_mask);

            if(work_queue != last_work_queue){
                lock = Lock_t<warp_mutex>();
            }
        }
    }

};
}
}
