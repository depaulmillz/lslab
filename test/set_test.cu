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
#include <unordered_map>
#include <stdexcept>
#include <lslab/set.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>

#define ASSERT(x, y) \
if (!(x)) { \
    throw std::runtime_error((y)); \
}

#define ASSERT_EQ(x, y, z) ASSERT(((x) == (y)), (z))
#define ASSERT_NE(x, y, z) ASSERT(((x) != (y)), (z))

using namespace lslab;

void ContainsInsertRemoveTest() {

    constexpr int size = 1024;

    set<int> m(10);

    for (int rep = 0; rep < 100; rep++) {
        thrust::device_vector<int> keys(size);
        thrust::device_vector<bool> res1(size);
        thrust::device_vector<bool> res2(size);
        thrust::sequence(keys.begin(), keys.end(), 1);

        std::cout << "Putting" << std::endl;

        m.insert(keys, res1);
       
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Done" << std::endl;

        m.contains(keys, res2);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Done" << std::endl;

        thrust::host_vector<bool> res_ = res2;
        for(auto b : res_) {
            ASSERT(b, "");
        }

        m.remove(keys, res1);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Done" << std::endl;

        m.contains(keys, res2);
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Done" << std::endl;

        thrust::host_vector<bool> res2_ = res2;
        for(auto b : res2_) {
            ASSERT(!b, "");
        }
    }
}

int main() {
    ContainsInsertRemoveTest();
    return 0;
}
