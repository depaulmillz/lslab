/**
 * @file
 */
#include "lslab.h"

#pragma once

namespace lslab {

template<typename T>
struct hash {
    LSLAB_HOST_DEVICE size_t operator()(T x) {
        return static_cast<size_t>(x);
    }
};


};

