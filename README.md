# LSlab

Designed in "KVCG: A Heterogeneous Key-Value Store for Skewed Workloads" by dePaul Miller, Jacob Nelson, Ahmed Hassan, and Roberto Palmieri."

## Building

Requires CMake >= 3.18, Conan, and a CUDA version in 11.0 to 11.6.

We also require [UnifiedMemoryGroupAllocation](https://github.com/depaulmillz/UnifiedMemoryGroupAllocation) built through conan.

Get CMake from kitware and Conan from your favorite python package manager.

It is easiest to get conan from pip by running
```
pip install conan
```

[Install CMake from Here!](https://cmake.org)

Next make a build directory and install with conan, and then build.
```
mkdir build
cd build
conan install --build missing ..
conan build ..
```

## Conan Options

- cuda\_arch is an option to specify the SM architecture you want to compile for, by default we compile for sm70 to sm86
- cuda\_compiler is an option to specify the CUDA compiler for example nvcc or clang++-13


## Code Organization

- include/LSlab contains all of the LSlab code
- LSlab.h contains basic macros
- LSlabMap.h contains a GPU interface for the map
- Operations.h contains code for setting up lslab on the host
- OperationsDevice.h contains code for using lslab on the device
- Slab.h contains basic structures for the host
- StandardSlabDefintions.h contains definitions used by lslab
- gpumemory.h contains GPU memory management functions.
- stdrequestHandler.cuh contains kernels
- test/ contains tests

## Clang Support

Clang seems to have an issue compiling the code that makes the tests fail.
I am not going to fix this as of this moment unless necessary.

