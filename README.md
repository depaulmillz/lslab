# lslab

Designed in "KVCG: A Heterogeneous Key-Value Store for Skewed Workloads" by dePaul Miller, Jacob Nelson, Ahmed Hassan, and Roberto Palmieri.

### Hardware

Runs on Volta, sm\_70, or greater.

### Building

Requires CMake >= 3.18, Conan, and a CUDA version in 11.0 to 12.1.

We also require [UnifiedMemoryGroupAllocation](https://github.com/depaulmillz/UnifiedMemoryGroupAllocation) built through conan.

Get CMake from kitware and Conan from your favorite python package manager.

It is easiest to get conan from pip by running
```
pip install conan==1.58
```

[Install CMake from Here!](https://cmake.org)

Next make a build directory and install with conan, and then build.
```
mkdir build
cd build
conan install --build missing ..
conan build ..
```

### Conan Options

- cuda\_arch is an option to specify the SM architecture you want to compile for, by default we compile for sm70 to sm90
- cuda\_compiler is an option to specify the CUDA compiler for example nvcc

### Code Organization

- include/lslab contains all of the lslab code
    - lslab.h contains basic macros
    - map.h contains a GPU interface for the map
    - hash.h contains GPU hash functions
    - device\_allocator.h contains a device allocator
    - mutex.h contains a mutex implementation
    - set.h contains the set implementation
    - warp\_mutex.h contains a cooperative warp mutex
    - detail contains extra implementation details
- test contains tests
- benchmark contains benchmarks

## Clang Support

Clang seems to have an issue compiling the code that makes the tests fail.
I am not going to fix this as of this moment unless necessary.

