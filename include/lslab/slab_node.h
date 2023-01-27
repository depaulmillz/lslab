#include "lslab.h"
#include <atomic>
#include <cub/util_ptx.cuh>
#include <type_traits>
#include <cuda/atomic>

#pragma once

namespace lslab {

template<typename K_, typename V_>
struct slab_node {
    using K = K_; //typename std::conditional<sizeof(K_) < sizeof(uint64_t), uint64_t, K_>::type;
    using V = V_;
    //static_assert(alignof(K) % alignof(void*) == 0, "Alignment must be a multiple of pointer alignment"); 

    // whether it is empty or not
    alignas(32) uint32_t valid = 0;    
    K key[31];
    slab_node<K, V>* next;
    V value[32];
};    

}
