#include "lslab.h"
#include <atomic>
#include <cuda/atomic>

#pragma once

namespace lslab {

struct mutex {

    mutex() : l(0) {}

    LSLAB_DEVICE void lock() {
        int expect = 0;
        while(!l.compare_exchange_strong(expect,-1)) {
            expect = 0;
        }
    }

    LSLAB_DEVICE void unlock() {
        l.store(0);
    }
   
    union {
        alignas(32) char _[32];
        cuda::std::atomic<int> l;
    }; 
};

}
