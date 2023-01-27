/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include "testheader.h"
#include <unordered_map>
#include <stdexcept>

#define ASSERT(x, y) \
if (!(x)) { \
    throw std::runtime_error((y)); \
}

#define ASSERT_EQ(x, y, z) ASSERT(((x) == (y)), (z))
#define ASSERT_NE(x, y, z) ASSERT(((x) != (y)), (z))

using namespace lslab;

const int BLOCKS = 128;
const int THREADS_PER_BLOCK = 512;

void MemoryLeakageTest() {

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

    for (int rep = 0; rep < 100; rep++) {

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

    delete b;
}

void GetPutTest() {

    const int size = 1000;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = j;
                int *value = new int[256]; // allocating 1KB
                for (int w = 0; w < 256; w++) {
                    value[w] = rep;
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

                    ASSERT(b->getBatchValues()[j] != nullptr, "GetPutTest fail");
                    for (int w = 0; w < 256; w++) {
                        ASSERT(b->getBatchValues()[j][w] == rep - 1, "GetPutTest fail old insert was rep - 1");
                    }

                    delete[] b->getBatchValues()[j];
                }
            }
        }

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = j;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_GET;
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
                if (b->getBatchRequests()[j] == REQUEST_GET) {
                    ASSERT(b->getBatchValues()[j] != nullptr, "");
                    for (int w = 0; w < 256; w++) {
                        ASSERT(b->getBatchValues()[j][w] == rep, " last insert was rep");
                    }
                }
            }
        }
    }

    delete b;
}

void PutRemoveTest() {

    const int size = 1000;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = j;
                int *value = new int[256]; // allocating 1KB
                for (int w = 0; w < 256; w++) {
                    value[w] = rep;
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
                if (b->getBatchRequests()[j] == REQUEST_INSERT) {
                    ASSERT_EQ(b->getBatchValues()[j], nullptr, " should always be reading nullptr last");
                }
            }
        }

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = j;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_REMOVE;
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
                if (b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    ASSERT_NE(b->getBatchValues()[j], nullptr, " key value pair was inserted on key");
                    for (int w = 0; w < 256; w++) {
                        ASSERT_EQ(b->getBatchValues()[j][w], rep, " last insert was rep");
                    }
                    delete[] b->getBatchValues()[j];
                }
            }
        }
    }

    delete b;
}

void PutRemoveTest_uint64() {


    const int size = 1000;
    std::hash<unsigned long long> hfn;
    SlabUnified<unsigned long long, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned long long, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned long long key = j;
                int *value = new int[256]; // allocating 1KB
                for (int w = 0; w < 256; w++) {
                    value[w] = rep;
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
            gpuErrchk(cudaPeekAtLastError());
            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT) {
                    ASSERT_EQ(b->getBatchValues()[j], nullptr, " should always be reading nullptr last");
                }
            }
        }

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                unsigned key = j;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_REMOVE;
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
                if (b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    ASSERT_NE(b->getBatchValues()[j], nullptr, " key value pair was inserted on key");
                    for (int w = 0; w < 256; w++) {
                        ASSERT_EQ(b->getBatchValues()[j][w], rep, " last insert was rep");
                    }
                    delete[] b->getBatchValues()[j];
                }
            }
        }
    }

    delete b;
}

LSLAB_HOST_DEVICE int memcmp_(const void* a, const void* b, size_t size) noexcept {
    for(size_t i = 0; i < size; i++) {
        char diff = reinterpret_cast<const char*>(a)[i] - reinterpret_cast<const char*>(b)[i];
        if(diff != 0) return diff;
    }
    return 0;
}

struct Key {

    LSLAB_HOST_DEVICE constexpr Key() : bytes{0} {}

    LSLAB_HOST_DEVICE Key(const Key& k) {
        memcpy(bytes, k.bytes, sizeof(bytes));
    }

    LSLAB_HOST_DEVICE Key(unsigned long long i) {
        memset(bytes, 0, sizeof(bytes));
        memcpy(bytes, &i, sizeof(unsigned long long));
    }

    LSLAB_HOST_DEVICE bool operator==(const Key k) const {
        return memcmp_(bytes, k.bytes, 128) == 0;
    }

    friend std::ostream& operator<<(std::ostream&, const Key&);

    alignas(128) char bytes[128];
};

std::ostream& operator<<(std::ostream& s, const Key& k) {
    unsigned long long i = 0;
    memcpy(&i, k.bytes, sizeof(unsigned long long));
    s << i;
    return s;
}

namespace std {

template<>
struct hash<Key> {

    std::size_t operator()(const Key& k) const {
        return k.bytes[0];
    }

};

}

void PutRemoveTest_128B() {


    const int BLOCKS_ = 128;
    const int CHOSEN_THREADS_PER_BLOCK = 32;

    const int size = 1000;
    std::hash<Key> hfn;
    SlabUnified<Key, int *, BLOCKS_, CHOSEN_THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<Key, int *, BLOCKS_, CHOSEN_THREADS_PER_BLOCK>();

    s.setGPU();

    std::unordered_map<Key, int*> reference;
    std::unordered_map<int*, Key> reverse;

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += CHOSEN_THREADS_PER_BLOCK * BLOCKS_) {
            unsigned j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_ && i * CHOSEN_THREADS_PER_BLOCK * BLOCKS_ + j < size; j++) {
                unsigned long long key = j + 1;
                int *value = new int[256]; // allocating 1KB
                reference[key] = value;
                reverse[value] = key;
                //std::cerr << "(" << key << "," << (void*)value << ")" << std::endl;
                for (int w = 0; w < 256; w++) {
                    value[w] = rep;
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_INSERT;
                b->getBatchValues()[j] = value;
                value = nullptr;
            }
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
                b->getBatchValues()[j] = nullptr;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS_, CHOSEN_THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            gpuErrchk(cudaPeekAtLastError());
            j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT) {
                    ASSERT_EQ(b->getBatchValues()[j], nullptr, " should always be reading nullptr last. Found incorrect at ");

                }
            }
        }

        for (unsigned i = 0; i < (unsigned) size; i += CHOSEN_THREADS_PER_BLOCK * BLOCKS_) {
            unsigned j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_ && i * CHOSEN_THREADS_PER_BLOCK * BLOCKS_ + j < size; j++) {
                unsigned key = j + 1;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_REMOVE;
                b->getBatchValues()[j] = nullptr; // to catch errors
            }
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
                b->getBatchValues()[j] = nullptr; // to catch errors
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS_, CHOSEN_THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            gpuErrchk(cudaPeekAtLastError());

            j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                if (b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    ASSERT_NE(b->getBatchValues()[j], nullptr, "");
                    ASSERT_EQ(b->getBatchValues()[j], reference[b->getBatchKeys()[j]], "");
                    for (int w = 0; w < 256; w++) {
                        ASSERT_EQ(b->getBatchValues()[j][w], rep, "");
                    }
                    delete[] b->getBatchValues()[j];
                    b->getBatchValues()[j] = nullptr;
                }
            }
        }
    }

    delete b;
}

void PutRemoveTest_128Bto128B() {


    const int BLOCKS_ = 10;
    const int CHOSEN_THREADS_PER_BLOCK = 32;

    const int size = 1000;
    std::hash<Key> hfn;
    SlabUnified<Key, Key, BLOCKS_, CHOSEN_THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<Key, Key, BLOCKS_, CHOSEN_THREADS_PER_BLOCK>();

    s.setGPU();

    std::unordered_map<Key, Key> reference;
    std::unordered_map<Key, Key> reverse;

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += CHOSEN_THREADS_PER_BLOCK * BLOCKS_) {
            unsigned j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_ && i * CHOSEN_THREADS_PER_BLOCK * BLOCKS_ + j < size; j++) {
                unsigned long long key = j + 1;
                unsigned long long value = rep;
                reference[key] = value;
                reverse[value] = key;
                //std::cerr << "(" << key << "," << (void*)value << ")" << std::endl;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_INSERT;
                b->getBatchValues()[j] = value;
            }
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS_, CHOSEN_THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            gpuErrchk(cudaPeekAtLastError());
            j = 0;
        }

        for (unsigned i = 0; i < (unsigned) size; i += CHOSEN_THREADS_PER_BLOCK * BLOCKS_) {
            unsigned j = 0;
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_ && i * CHOSEN_THREADS_PER_BLOCK * BLOCKS_ + j < size; j++) {
                unsigned key = j + 1;
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchRequests()[j] = REQUEST_REMOVE;
                b->getBatchValues()[j] = 0; // to catch errors
            }
            for (; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
                b->getBatchValues()[j] = 0; // to catch errors
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS_, CHOSEN_THREADS_PER_BLOCK, 0x0);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            gpuErrchk(cudaPeekAtLastError());

            j = 0;
        }
    }

    delete b;
}

int main() {

    MemoryLeakageTest();
    GetPutTest();
    PutRemoveTest();
    PutRemoveTest_uint64();
    PutRemoveTest_128B();
    PutRemoveTest_128Bto128B();
    return 0;
}
