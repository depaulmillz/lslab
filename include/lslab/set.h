/**
 * @file
 */
#include "detail/slab_node.h"
#include "detail/traverse.h"
#include "device_allocator.h"
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda/std/utility>
#include "hash.h"
#include "warp_mutex.h"
#include "detail/set_kernels.h"

#pragma once

namespace lslab {

/**
 * LSlab set for GPU
 */
template<typename K, typename Allocator = device_allocator<detail::set_node<K>>, typename Hash = hash<K>>
class set {
public:

    using this_t = set<K, Allocator, Hash>;

    LSLAB_HOST set() : set(10) {
        
    }

    LSLAB_HOST set(unsigned n_log_2) : number_of_buckets_log_2(n_log_2) {
        size_t size = 1 << n_log_2;
        cudaMalloc(&lock_table, sizeof(warp_mutex) * size);

        cudaMemset(lock_table, 0, sizeof(warp_mutex) * size);

        cudaMalloc(&buckets_array, sizeof(detail::set_node<K>) * size);
        
        cudaMemset(buckets_array, 0, sizeof(detail::set_node<K>) * size);
    }

    LSLAB_HOST set(unsigned n_log_2, Allocator&& a) : number_of_buckets_log_2(n_log_2), alloc(a) {
        size_t size = 1 << n_log_2;
        cudaMalloc(&lock_table, sizeof(warp_mutex) * size);

        cudaMemset(lock_table, 0, sizeof(warp_mutex) * size);

        cudaMalloc(&buckets_array, sizeof(detail::set_node<K>) * size);
        
        cudaMemset(buckets_array, 0, sizeof(detail::set_node<K>) * size);
    }
    
    LSLAB_HOST_DEVICE set(warp_mutex* lt, detail::set_node<K>* s, unsigned n_log_2) : lock_table(lt), buckets_array(s), number_of_buckets_log_2(n_log_2) {

    }

    LSLAB_DEVICE set(warp_mutex* lt, detail::set_node<K>* s, unsigned n_log_2, Allocator&& a) : lock_table(lt), buckets_array(s), number_of_buckets_log_2(n_log_2), alloc(a) {

    }

    LSLAB_HOST_DEVICE ~set() {
    } 

    LSLAB_DEVICE bool contains(const K& key, bool thread_mask = true) {

        size_t hash = Hash{}(key);
        hash &= ((1 << number_of_buckets_log_2) - 1);

        bool result = false;
        detail::traverse<Allocator, detail::OPERATION_TYPE::FIND>{}(lock_table, buckets_array, key, result, alloc, hash, thread_mask); 
        return result;
    }

    LSLAB_DEVICE bool insert(const K& key, bool thread_mask = true) {
        using traverse_t = detail::traverse<Allocator, detail::OPERATION_TYPE::INSERT>;
        traverse_t t;
        size_t hash = Hash{}(key) & ((1 << number_of_buckets_log_2) - 1);
        bool result = false;
        t.template operator()<K>(lock_table, buckets_array, key, result, alloc, hash, thread_mask); 
        return result;
    }

    LSLAB_DEVICE bool remove(const K& key, bool thread_mask = true) {
        using traverse_t = detail::traverse<Allocator, detail::OPERATION_TYPE::REMOVE>;
        traverse_t t;
        size_t hash = Hash{}(key) & ((1 << number_of_buckets_log_2) - 1);
        bool result = false;
        t.template operator()<K>(lock_table, buckets_array, key, result, alloc, hash, thread_mask); 
        return result;
    }

    template<int block_size = 256>
    LSLAB_HOST void contains(const K* keys, bool* output, size_t size, cudaStream_t stream = 0x0) {
        set_kernels::contains_<block_size, this_t, K><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, keys, output, size);
    }

    template<int block_size = 256>
    LSLAB_HOST void contains(const thrust::device_vector<K>& keys, thrust::device_vector<bool>& output, cudaStream_t stream = 0x0) {
        size_t size = keys.size();
        this->template contains<block_size>(keys.data().get(), output.data().get(), size, stream);
    }
    
    template<int block_size = 256>
    LSLAB_HOST void insert(const K* keys, bool* output, size_t size, cudaStream_t stream = 0x0) {
        set_kernels::insert_<block_size, this_t, K><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, keys, output, size);
    }

    template<int block_size = 256>
    LSLAB_HOST void insert(const thrust::device_vector<K>& keys, thrust::device_vector<bool>& output, cudaStream_t stream = 0x0) {
        size_t size = keys.size();
        this->template insert<block_size>(keys.data().get(), output.data().get(), size, stream);
    }

    template<int block_size = 256>
    LSLAB_HOST void remove(const K* keys, bool* output, size_t size, cudaStream_t stream = 0x0) {
        set_kernels::remove_<block_size, this_t, K><<<(size + block_size - 1) / block_size, block_size, 0, stream>>>(*this, keys, output, size);
    }

    template<int block_size = 256>
    LSLAB_HOST void remove(const thrust::device_vector<K>& keys, thrust::device_vector<bool>& output, cudaStream_t stream = 0x0) {
        size_t size = keys.size();
        this->template remove<block_size>(keys.data().get(), output.data().get(), size, stream);
    }

    LSLAB_HOST_DEVICE unsigned buckets() const {
        return 1 << number_of_buckets_log_2;
    }

private:
    warp_mutex* lock_table;
    detail::set_node<K>* buckets_array;
    unsigned number_of_buckets_log_2;
    Allocator alloc;
};

}
