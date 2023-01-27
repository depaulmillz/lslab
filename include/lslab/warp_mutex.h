#include "lslab.h"
#include <atomic>
#include <cuda/atomic>

#pragma once

namespace lslab {

struct warp_mutex {

    warp_mutex() : l(0) {}

    LSLAB_DEVICE void lock() {
        if(threadIdx.x % 32 == 0) {
            int expect = 0;
            while(!l.compare_exchange_strong(expect,-1)) {
                expect = 0;
            }
        }
        __syncwarp();
    }

    LSLAB_DEVICE void unlock() {
        if(threadIdx.x % 32 == 0) {
            l.store(0);
        }
    }
    
    LSLAB_DEVICE void shared_lock() {
        if(threadIdx.x % 32 == 0) {
            while(true) {
                int pred = l.load();
                if(pred != -1 && l.compare_exchange_strong(pred, pred + 1)) {
                    break;
                }
            }
        }
        __syncwarp();
    }

    LSLAB_DEVICE int status() {
        return l.load();
    }

    LSLAB_DEVICE void shared_unlock() {
        if(threadIdx.x % 32 == 0) {
            l.fetch_add(-1);
        }
    }

    union {
        alignas(32) char _[32];
        cuda::std::atomic<int> l;
    }; 
};

template<typename T>
struct warp_unique_lock {
    LSLAB_DEVICE warp_unique_lock() : mtx(nullptr) {}

    LSLAB_DEVICE warp_unique_lock(T& mtx_) : mtx(&mtx_) {
        mtx->lock();
    }
    
    LSLAB_DEVICE warp_unique_lock(const warp_unique_lock<T>&) = delete;

    LSLAB_DEVICE warp_unique_lock(warp_unique_lock<T>&& other) {
        mtx = other.mtx;
        other.mtx = nullptr;
    }

    LSLAB_DEVICE warp_unique_lock& operator=(warp_unique_lock<T>&& other) {
        if(mtx) mtx->unlock();
        mtx = other.mtx;
        other.mtx = nullptr;
    }

    LSLAB_DEVICE ~warp_unique_lock() {
        if(mtx) mtx->unlock();
    }

    T* mtx;
};

template<typename T>
struct warp_shared_lock {

    LSLAB_DEVICE warp_shared_lock() : mtx(nullptr) {}

    LSLAB_DEVICE warp_shared_lock(const warp_shared_lock<T>&) = delete;

    LSLAB_DEVICE warp_shared_lock(warp_shared_lock<T>&& other) {
        mtx = other.mtx;
        other.mtx = nullptr;
    }

    LSLAB_DEVICE warp_shared_lock& operator=(warp_shared_lock<T>&& other) {
        if(mtx) mtx->shared_unlock();
        mtx = other.mtx;
        other.mtx = nullptr;
    }

    LSLAB_DEVICE warp_shared_lock(T& mtx_) : mtx(&mtx_) {
        mtx->shared_lock();
    }

    LSLAB_DEVICE ~warp_shared_lock() {
        if(mtx) mtx->shared_unlock();
    }

    T* mtx;
};

template<typename T>
struct warp_noop_lock {

    LSLAB_DEVICE warp_noop_lock() : mtx(nullptr) {}

    LSLAB_DEVICE warp_noop_lock(const warp_noop_lock<T>&) = delete;

    LSLAB_DEVICE warp_noop_lock(warp_noop_lock<T>&& other) {
    }

    LSLAB_DEVICE warp_noop_lock& operator=(warp_noop_lock<T>&& other) {
    }

    LSLAB_DEVICE warp_noop_lock(T& mtx_) : mtx(&mtx_) {
    }

    LSLAB_DEVICE ~warp_noop_lock() {
    }

    T* mtx;
};

}
