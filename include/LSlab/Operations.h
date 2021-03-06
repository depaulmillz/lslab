/**
 * @file
 */
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpumemory.h"
#include <iostream>
#include "OperationsDevice.h"
#include <GroupAllocator/groupallocator>

#pragma once

namespace lslab {

template<typename K, typename V>
std::ostream &operator<<(std::ostream &output, const SlabData<K, V> &s) {
    output << s.keyValue;
    return output;
}

/// Creates warp allocator context to be able to allocate slabs on GPU
template<typename K, typename V>
WarpAllocCtx<K, V>
setupWarpAllocCtxGroup(groupallocator::GroupAllocator &gAlloc, int threadsPerBlock, int blocks, int gpuid = 0,
                       cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid))
    WarpAllocCtx<K, V> actx;
    gAlloc.allocate(&actx.blocks, (size_t) ceil(threadsPerBlock * blocks / 32.0) * sizeof(SuperBlock<K, V>), false);
    for (size_t i = 0; i < (size_t) ceil(threadsPerBlock * blocks / 32.0); i++) {
        gAlloc.allocate(&(actx.blocks[i].memblocks), sizeof(MemoryBlock<K, V>) * 32, false);
        for (int j = 0; j < 32; j++) {
            actx.blocks[i].memblocks[j].bitmap = ~0ull;
            gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab), sizeof(SlabData<K, V>) * 64, false);
            for (int k = 0; k < 64; k++) {
                //gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab[k].keyValue), sizeof(unsigned long long) * 32, false);
                actx.blocks[i].memblocks[j].slab[k].ilock = 0;
                for (int w = 0; w < 32; w++) {
                    actx.blocks[i].memblocks[j].slab[k].key[w] = K{};
                    actx.blocks[i].memblocks[j].slab[k].value[j] = V{};
                }
            }
        }
    }
    gAlloc.moveToDevice(gpuid, stream);
    gpuErrchk(cudaDeviceSynchronize())
    std::cerr << "Size allocated for warp alloc: "
              << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
              << std::endl;
    return actx;
}

/// Sets up LSlab using the group allocator
template<typename K, typename V>
LSLAB_HOST SlabCtx<K, V> *setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid = 0,
                          cudaStream_t stream = cudaStreamDefault) {

    gpuErrchk(cudaSetDevice(gpuid));

    auto sctx = new SlabCtx<K, V>();
    sctx->num_of_buckets = size;
    std::cerr << "Size of index is " << size << std::endl;
    std::cerr << "Each slab is " << sizeof(SlabData<K, V>) << "B" << std::endl;


    gAlloc.allocate(&(sctx->slabs), sizeof(void *) * sctx->num_of_buckets, false);

    for (int i = 0; i < sctx->num_of_buckets; i++) {
        gAlloc.allocate(&(sctx->slabs[i]), sizeof(SlabData<K, V>), false);

        static_assert(sizeof(sctx->slabs[i]->key[0]) >= sizeof(void *),
                      "The key size needs to be greater or equal to the size of a memory address");

        //gAlloc.allocate((unsigned long long **) &(sctx->slabs[i][k].keyValue), sizeof(unsigned long long) * 32, false);

        memset((void *) (sctx->slabs[i]), 0, sizeof(SlabData<K, V>));

        for (int j = 0; j < 31; j++) {
            const_cast<typename SlabData<K, V>::KSub *>(sctx->slabs[i]->key)[j] = K{};// EMPTY_PAIR;
        }

        void **ptrs = (void **) sctx->slabs[i]->key;

        ptrs[31] = nullptr;// EMPTY_POINTER;

        for (int j = 0; j < 32; j++) {
            const_cast<V&>(sctx->slabs[i]->value[j]) = V{};
        }

    }

    gAlloc.moveToDevice(gpuid, stream);

    gpuErrchk(cudaDeviceSynchronize())

    std::cerr << "Size allocated for Slab: "
              << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
              << std::endl;
    return sctx;
}

}
