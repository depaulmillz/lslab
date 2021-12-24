//
// Created by depaulsmiller on 7/24/20.
//

#include "LSlab.h"
#include "Operations.h"
#include "LSlabMap.h"

#pragma once

namespace lslab {

const int REQUEST_INSERT = 1;
const int REQUEST_GET = 2;
const int REQUEST_REMOVE = 3;
const int REQUEST_EMPTY = 0;

static_assert(Operation::GET == REQUEST_GET, "Need Operations to be Same");
static_assert(Operation::PUT == REQUEST_INSERT, "Need Operations to be Same");
static_assert(Operation::REMOVE == REQUEST_REMOVE, "Need Operations to be Same");
static_assert(Operation::NOP == REQUEST_EMPTY, "Need Operations to be Same");

/**
 * mvValue index is set to the value on a GET or EMPTY<V>::value if there is no value
 * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
 * @tparam K
 * @tparam V
 * @param slabs
 * @param num_of_buckets
 * @param myKey
 * @param myValue
 * @param myHash
 * @param request
 * @param ctx
 */
template<typename K, typename V>
__global__ void requestHandler(volatile SlabData<K, V> **slabs, unsigned num_of_buckets,
                               K *myKey,
                               V *myValue, const unsigned * myHash, const int *request, WarpAllocCtx<K, V> ctx) {
    const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    LSlab<K,V> l{slabs, num_of_buckets, ctx};

    K key = myKey[tid];
    V value = myValue[tid];
    unsigned hash = myHash[tid] % num_of_buckets;
    bool activity = (request[tid] == REQUEST_GET);

    l.get(key, value, hash, activity);

    activity = (request[tid] == REQUEST_INSERT || request[tid] == REQUEST_REMOVE);
    Operation req = static_cast<Operation>(request[tid]);
    l.modify(key, value, hash, req, activity);

    myValue[tid] = value;
}


/**
 * mvValue index is set to the value on a GET or EMPTY<V>::value if there is no value
 * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
 * @tparam K
 * @tparam V
 * @param slabs
 * @param num_of_buckets
 * @param myKey
 * @param myValue
 * @param myHash
 * @param request
 * @param ctx
 */
template<typename K, typename V>
__global__ void requestHandlerAPI(volatile SlabData<K, V> **slabs, unsigned num_of_buckets,
                               K *myKey,
                               V *myValue, const unsigned * myHash, const int *request, WarpAllocCtx<K, V> ctx) {
    const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    LSlab<K,V> l{slabs, num_of_buckets, ctx};

    K key = myKey[tid];
    V value = myValue[tid];
    unsigned hash = myHash[tid] % num_of_buckets;
    bool activity = (request[tid] == REQUEST_GET);

    l.get(key, value, hash, activity);

    activity = (request[tid] == REQUEST_INSERT);
    l.put(key, value, hash, activity);

    activity = (request[tid] == REQUEST_REMOVE);
    l.remove(key, value, hash, activity);
    myValue[tid] = value;
}

/**
 * mvValue index is set to the value on a GET or EMPTY<V>::value if there is no value
 * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
 * @tparam K
 * @tparam V
 * @param slabs
 * @param num_of_buckets
 * @param myKey
 * @param myValue
 * @param myHash
 * @param request
 * @param ctx
 */
template<typename K, typename V>
__global__ void requestHandlerTraditional(volatile SlabData<K, V> **slabs, unsigned num_of_buckets,
                               K *myKey,
                               V *myValue, const unsigned * myHash, const int *request, WarpAllocCtx<K, V> ctx) {
    const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

    K key = myKey[tid];
    V value = myValue[tid];
    unsigned hash = myHash[tid] % num_of_buckets;
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, hash, slabs, num_of_buckets);

    activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, hash, slabs,
                           num_of_buckets, ctx);

    activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, hash, slabs, num_of_buckets);
    myValue[tid] = value;
}

}
