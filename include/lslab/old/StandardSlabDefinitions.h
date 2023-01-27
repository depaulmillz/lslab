/**
 * @file
 */
#include <functional>
#include <string>
#include <LSlab/LSlab.h>

#pragma once

namespace lslab {

/**
 * A type to store a chunk of data of the given size
 */
struct data_t {

    /// Creates an empty data_t
    data_t() : size(0), data(nullptr) {}

    /// Allocates a data_t of size s
    data_t(size_t s) : size(s), data(new char[s]) {}

    /// Note this doesn't free the underlying data
    ~data_t() {}

    size_t size;
    char *data;

    /// Shallow copy of the data_t
    data_t &operator=(const data_t &rhs) {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

    /// Shallow copy of the data_t
    volatile data_t &operator=(const data_t &rhs) volatile {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

    /// Checks if the size is the same and them memcmp
    LSLAB_HOST_DEVICE bool operator==(const data_t& other) const {
        if (size != other.size) {
            return false;
        }

        return memcmp(data, other.data, size) == 0;
    }

};

/// For use with shared_ptr
class Data_tDeleter{
    void operator()(data_t* ptr) const noexcept {
        delete[] ptr->data;
        delete ptr;
    }
};

}

namespace std {
    template<>
    struct hash<lslab::data_t *> {
        std::size_t operator()(lslab::data_t *&x) {
            return std::hash<std::string>{}(x->data) ^ std::hash<std::size_t>{}(x->size);
        }
    };
}


