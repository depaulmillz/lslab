//
// Created by depaulsmiller on 9/3/20.
//

#include <LSlab/StandardSlabDefinitions.h>
#include <vector>
#include <LSlab/Slab.h>
#include <cuda_profiler_api.h>
#include <unordered_map>

using namespace lslab;

const int BLOCKS = 128;
const int THREADS_PER_BLOCK = 512;

int main() {

    const int size = 1000;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
        unsigned j = 0;
        for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
            unsigned key = 1;
            int *value = new int[256]; // allocating 1KB
            for (int w = 0; w < 256; w++) {
                value[w] = 1;
            }
            b->getBatchKeys()[j] = key;
            b->getHashValues()[j] = hfn(key);
            b->getBatchRequests()[j] = REQUEST_INSERT;
            b->getBatchValues()[j] = value;
        }
        for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
            b->getBatchRequests()[j] = REQUEST_EMPTY;
        }
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        j = 0;
        for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
            if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != nullptr) {
                delete[] b->getBatchValues()[j];
            }
        }
    }

    gpuErrchk(cudaProfilerStart());

    for (int rep = 0; rep < 10; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = 1;
                int *value = new int[256]; // allocating 1KB
                for (int w = 0; w < 256; w++) {
                    value[w] = 1;
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_INSERT;
                b->getBatchValues()[j] = value;
            }
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));

            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != nullptr) {
                    delete[] b->getBatchValues()[j];
                }
            }
        }
    }

    gpuErrchk(cudaProfilerStop());
    delete b;
}
