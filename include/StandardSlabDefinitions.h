//
// Created by depaulsmiller on 8/28/20.
//
#include <functional>
#include "ImportantDefinitions.h"

#pragma once

namespace lslab {

struct data_t {

    data_t() : size(0), data(nullptr) {}

    data_t(size_t s) : size(s), data(new char[s]) {}

    /// Note this doesn't free the underlying data
    ~data_t() {}

    size_t size;
    char *data;

    data_t &operator=(const data_t &rhs) {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

    volatile data_t &operator=(const data_t &rhs) volatile {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

};

/// For use with shared_ptr
class Data_tDeleter{
    void operator()(data_t* ptr) const noexcept {
        delete[] ptr->data;
        delete ptr;
    }
};

template<>
struct EMPTY<data_t *> {
    static constexpr data_t *value = nullptr;
};

template<>
LSLAB_HOST_DEVICE unsigned compare(const data_t * lhs, const data_t * rhs) {

    if (lhs == rhs) {
        return 0;
    } else if (lhs == nullptr || rhs == nullptr) {
        return 1;
    }

    if (lhs->size != rhs->size) {
        return (unsigned) (lhs->size - rhs->size);
    }

    for (size_t i = 0; i < lhs->size; i++) {
        unsigned sub = lhs->data[i] - rhs->data[i];
        if (sub != 0)
            return sub;
    }

    return 0;
}

}

namespace std {
    template<>
    struct std::hash<lslab::data_t *> {
        std::size_t operator()(lslab::data_t *&x) {
            return std::hash<std::string>{}(x->data) ^ std::hash<std::size_t>{}(x->size);
        }
    };
}


