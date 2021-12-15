//
// Created by depaulsmiller on 9/23/20.
//
#include "LSlab.h"

#pragma once

namespace lslab {

template<typename T>
struct EMPTY {
    static constexpr T value{};
};

template<typename T>
LSLAB_HOST_DEVICE unsigned compare(T lhs, T rhs);

// Definitions for standard types

template<>
struct EMPTY<unsigned> {
    static const unsigned value = 0;
};

template<>
struct EMPTY<unsigned long long> {
    static const unsigned long long value = 0;
};


template<>
LSLAB_HOST_DEVICE unsigned compare(const unsigned lhs, const unsigned rhs) {
    return lhs - rhs;
}

template<>
LSLAB_HOST_DEVICE unsigned compare(const int lhs, const int rhs) {
    return lhs - rhs;
}

template<>
LSLAB_HOST_DEVICE unsigned compare(const unsigned long long lhs, const unsigned long long rhs) {
    return lhs - rhs;
}


}
