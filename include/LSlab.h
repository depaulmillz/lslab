#include <cstdio>

#pragma once

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))

#define LSLAB_HOST __host__

#define LSLAB_HOST_DEVICE __host__ __device__ __forceinline__

#define LSLAB_DEVICE __device__ __forceinline__

#else

#define LSLAB_HOST inline

#define LSLAB_DEVICE inline

#define LSLAB_HOST_DEVICE inline

#endif

#define gpuErrchk(ans)                                                         \
  { lslab::gpuAssert_slab((ans), __FILE__, __LINE__); }
namespace lslab {

LSLAB_HOST void gpuAssert_slab(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

}
