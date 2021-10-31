#include <cstdio>

#pragma once

#define gpuErrchk(ans)                                                         \
  { lslab::gpuAssert_slab((ans), __FILE__, __LINE__); }
namespace lslab {

inline void gpuAssert_slab(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

}
